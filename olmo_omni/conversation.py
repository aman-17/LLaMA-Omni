import dataclasses
from enum import Enum, auto
from typing import Any, List, Union


class SeparatorStyle(Enum):
    OLMO2 = auto()


@dataclasses.dataclass
class Conversation:
    """
    It is used to generate the prompt for the model.
    """

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.OLMO2
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    tokenizer_id: str = ""
    tokenizer: Any = None
    stop_str: Union[str, List[str]] = None
    stop_token_ids: List[int] = None
    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages

        if self.sep_style == SeparatorStyle.OLMO2:
            ret = "<|endoftext|>"
            if self.system:
                ret += f"<|system|>\n{self.system}\n"
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    if role == "user":
                        ret += f"<|user|>\n{message}\n"
                    elif role == "assistant":
                        if i == len(messages) - 1:
                            ret += f"<|assistant|>\n{message}<|endoftext|>"
                        else:
                            ret += f"<|assistant|>\n{message}<|endoftext|>\n"
                else:
                    if role == "assistant":
                        ret += f"<|assistant|>\n"

        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, speech = msg
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [
                    [x, y[0] if type(y) is tuple else y] for x, y in self.messages
                ],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_olmo2 = Conversation(
    system="You are a helpful language and speech assistant. "
    "You are able to understand the speech content that the user provides, "
    "and assist the user with a variety of tasks using natural language.",
    roles=("user", "assistant"),
    version="olmo2",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.OLMO2,
    sep="",
    sep2="",
)

default_conversation = conv_olmo2
conv_templates = {
    "olmo2": conv_olmo2,
}


def set_default_conversation(template_name):
    global default_conversation
    if template_name in conv_templates:
        default_conversation = conv_templates[template_name]
    else:
        raise ValueError(f"Unknown conversation template: {template_name}")


if __name__ == "__main__":
    print(default_conversation.get_prompt())
