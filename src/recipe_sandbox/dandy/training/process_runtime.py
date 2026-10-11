"""管理训练/评测子进程的生命周期，防止阶段结束后残留设备权重。

HPC/Linux中每个启动器使用独立进程组；正常结束、异常及SIGTERM均清理本组
派生进程，限时等待后强制退出并回收父进程。仅操作本模块创建的进程，不扫描
或终止其他用户任务。Windows本地检查只管理直接子进程，不宣称验证NPU释放。
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, Sequence


def _group_has_live_members(group_id: int) -> bool:
    """Linux只读检查本组的非僵尸成员；其他POSIX用killpg(0)确认组存在。

    僵尸不持有设备权重，不能因其等待init回收而误判清理失败。读取/proc遇到
    权限错误则直接报错，不把不可验证的资源释放当作成功。
    """
    if os.path.isdir("/proc"):
        for name in os.listdir("/proc"):
            if not name.isdecimal():
                continue
            try:
                with open("/proc/" + name + "/stat", encoding="utf-8") as stream:
                    fields = stream.read().rsplit(")", 1)[1].split()
            except (FileNotFoundError, ProcessLookupError):
                continue
            if int(fields[2]) == group_id and fields[0] not in {"Z", "X"}:
                return True
        return False
    try:
        os.killpg(group_id, 0)
        return True
    except ProcessLookupError:
        return False


def _stop_process(process: subprocess.Popen, grace_seconds: float = 10.0) -> None:
    """终止本模块创建的进程组，限时升级SIGKILL并wait回收直接子进程。

    POSIX按已保存的组ID操作，即使启动器先退出也能清理残留rank/vLLM worker。
    清理失败向上传播，阻止调用方继续启动下一阶段。
    """
    if os.name == "posix":
        def signal_group(sig: int) -> bool:
            """只向本次创建的进程组发信号；组不存在即已清理。"""
            try:
                os.killpg(process.pid, sig)
                return True
            except ProcessLookupError:
                return False

        alive = signal_group(signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        while alive and time.monotonic() < deadline:
            process.poll()  # 回收已退出的启动器，避免其僵尸状态阻塞组检查。
            alive = _group_has_live_members(process.pid)
            if alive:
                time.sleep(0.1)
        if alive:
            signal_group(signal.SIGKILL)
            deadline = time.monotonic() + grace_seconds
            while _group_has_live_members(process.pid):
                process.poll()
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"进程组{process.pid}强制终止后仍有活跃成员，停止后续阶段")
                time.sleep(0.1)
    elif process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
    process.wait(timeout=grace_seconds)


@contextmanager
def managed_process(command: Sequence[str], **kwargs: Any) -> Iterator[subprocess.Popen]:
    """启动独立进程组，退出上下文时清理全部组成员并恢复SIGTERM处理器。

    command与kwargs交给Popen；调用方负责等待并检查返回码。主线程中把SIGTERM
    转为SystemExit以执行finally。正常路径的清理错误会报错；异常路径附加清理
    错误而保留主异常。不支持PIPE输出（调用方应使用文件或继承终端）。
    """
    if kwargs.get("stdout") == subprocess.PIPE or kwargs.get("stderr") == subprocess.PIPE:
        raise ValueError("managed_process输出必须使用日志文件或继承终端")
    previous = None
    process = None
    primary = None
    main_thread = threading.current_thread() is threading.main_thread()

    def interrupted(signum: int, frame: Any) -> None:
        """保留非零信号退出语义，同时让上下文释放设备进程。"""
        raise SystemExit(128 + signum)

    if main_thread:
        previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        process = subprocess.Popen(command, start_new_session=(os.name == "posix"), **kwargs)
        yield process
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            if main_thread:
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
            if process is not None:
                try:
                    _stop_process(process)
                except BaseException as error:
                    if primary is None:
                        raise
                    message = "子进程清理失败: " + repr(error)
                    if hasattr(primary, "add_note"):
                        primary.add_note(message)
                    try:
                        print(message, file=sys.stderr, flush=True)
                    except Exception:
                        pass  # 日志关闭也不能覆盖原始失败。
        finally:
            if main_thread:
                signal.signal(signal.SIGTERM, previous)


def run_managed_process(command: Sequence[str], **kwargs: Any) -> None:
    """阻塞运行命令，非零退出抛CalledProcessError，返回前完成进程组清理。"""
    with managed_process(command, **kwargs) as process:
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
