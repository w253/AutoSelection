from enum import Enum


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TaskType(str, Enum):
    REASONING = "reasoning"
    FACTUAL_QA = "factual_qa"
    CODE = "code"
    CHAT = "chat"
    STYLE_IMITATION = "style_imitation"
    STRUCTURED_EXTRACTION = "structured_extraction"
    OTHER = "other"


class Split(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
    UNSPECIFIED = "unspecified"

    @classmethod
    def _missing_(cls, value: object) -> "Split | None":
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        aliases = {
            "training": cls.TRAIN,
            "train": cls.TRAIN,
            "validation": cls.DEV,
            "valid": cls.DEV,
            "val": cls.DEV,
            "development": cls.DEV,
            "dev": cls.DEV,
            "testing": cls.TEST,
            "test": cls.TEST,
            "unspecified": cls.UNSPECIFIED,
            "unknown": cls.UNSPECIFIED,
            "": cls.UNSPECIFIED,
        }
        return aliases.get(normalized)
