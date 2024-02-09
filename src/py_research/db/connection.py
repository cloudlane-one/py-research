"""Abstract base for connections to relational databases."""

from abc import ABC
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, overload

from yarl import URL

from .schema import (
    Attribute,
    AttrRef,
    AttrSet,
    Data,
    DataArray,
    DataBase,
    DataHub,
    DataSet,
    IndexLink,
    Schema,
)

S = TypeVar("S", bound="Schema")
S2 = TypeVar("S2", bound="Schema")
S3 = TypeVar("S3", bound="Schema")
S_cov = TypeVar("S_cov", covariant=True, bound="Schema")
S2_cov = TypeVar("S2_cov", covariant=True, bound="Schema")
S3_cov = TypeVar("S3_cov", covariant=True, bound="Schema")

A = TypeVar("A", bound="Attribute")
A_cov = TypeVar("A_cov", covariant=True, bound="Attribute")
A2_cov = TypeVar("A2_cov", covariant=True, bound="Attribute")

V = TypeVar("V")
V_cov = TypeVar("V_cov", covariant=True)

C_cov = TypeVar("C_cov", covariant=True, bound="Connection")
D_cov = TypeVar("D_cov", covariant=True, bound="Data")

Id = TypeVar("Id")
Id_cov = TypeVar("Id_cov", covariant=True, bound="Hashable")

L = TypeVar("L", bound="IndexLink")
L2 = TypeVar("L2", bound="IndexLink")


@dataclass(frozen=True, kw_only=True)
class Connection(ABC, Generic[C_cov, D_cov]):
    """Data connection."""

    spec: D_cov
    """Data."""

    container: C_cov
    """Index."""


@dataclass(frozen=True, kw_only=True)
class ArrayConnection(
    Connection[C_cov, DataArray[S_cov, V_cov, Id_cov]],
    ABC,
    Generic[C_cov, S_cov, V_cov, Id_cov],
):
    """TBD."""


@dataclass(frozen=True, kw_only=True)
class SetConnection(
    Connection[C_cov, DataSet[S_cov, V_cov, Id_cov, S2_cov, S3_cov]],
    ABC,
    Generic[C_cov, S_cov, V_cov, Id_cov, S2_cov, S3_cov],
):
    """TBD."""


@dataclass(frozen=True, kw_only=True)
class HubConnection(
    Connection[C_cov, DataHub[S_cov, V_cov, Id_cov, S2_cov, S3_cov, L]],
    ABC,
    Generic[C_cov, S_cov, V_cov, Id_cov, S2_cov, S3_cov, L],
):
    """TBD."""


@dataclass(frozen=True, kw_only=True)
class BaseConnection(
    Connection["BaseConnection", DataBase[S_cov, V_cov, Id_cov, S2_cov, S3_cov, L, L2]],
    ABC,
    Generic[C_cov, S_cov, V_cov, Id_cov, S2_cov, S3_cov, L, L2],
):
    """TBD."""

    url: URL
    namespace: Hashable | None = None

    @overload
    def __getitem__(
        self, key: AttrSet[S_cov, S, Any]
    ) -> "DataBase[S, Any, Any, Any, Any, Any, Any]": ...

    @overload
    def __getitem__(
        self, key: AttrSet[S2_cov, S2, Any]
    ) -> "DataBase[Any, S2, Any, Any, Any, Any, Any]": ...

    @overload
    def __getitem__(self, key: AttrRef[Data[S_cov, V], Any, S_cov]) -> V: ...

    @overload
    def __getitem__(
        self, key: AttrRef[Data[S2_cov, V], Any, S2_cov] | str
    ) -> V | None: ...

    # TODO: overloads for DataIndex, DataSet and DataHub
    # Requires implementation of Data*Connection instances above.

    def __getitem__(
        self,
        key: (
            AttrSet[S_cov | S2_cov, Any, Any]
            | AttrRef[Data[S_cov | S2_cov, V], Any, S_cov | S2_cov]
            | str
        ),
    ) -> "DataBase | V | None":
        """Get a column of this table."""
        ...
