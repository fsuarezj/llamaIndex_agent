from typing import Any
from pydantic import Field, PrivateAttr, BaseModel


class Foo(BaseModel):

    _foo_var: str = PrivateAttr(default="I'm foo")

    def __init__(self, addition: str, **kwargs: Any) -> None:
        self._foo_ = self._foo_var + addition

        super().__init__(
            **kwargs
        )

foo = Foo("Adding this")