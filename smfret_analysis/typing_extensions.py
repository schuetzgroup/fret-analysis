# Copyright (c) 2015, 2016, 2017, 2018, 2019 Python Software Foundation; All Rights Reserved
#
# SPDX-License-Identifier: Python-2.0
#
# From
# https://github.com/python/cpython/blob/master/Lib/typing.py
# https://github.com/python/typing/blob/master/src/typing.py
# https://github.com/python/typing/blob/master/typing_extensions/src_py3/typing_extensions.py

"""Backport typing.Literal to Python version < 3.8"""
import types


class TypingMeta(type):
    """Metaclass for most types defined in typing module
    (not a part of public API).
    This overrides __new__() to require an extra keyword parameter
    '_root', which serves as a guard against naive subclassing of the
    typing classes.  Any legitimate class defined using a metaclass
    derived from TypingMeta must pass _root=True.
    This also defines a dummy constructor (all the work for most typing
    constructs is done in __new__) and a nicer repr().
    """

    _is_protocol = False

    def __new__(cls, name, bases, namespace, *, _root=False):
        if not _root:
            raise TypeError("Cannot subclass %s" %
                            (', '.join(map(_type_repr, bases)) or '()'))
        return super().__new__(cls, name, bases, namespace)

    def __init__(self, *args, **kwds):
        pass

    def _eval_type(self, globalns, localns):
        """Override this in subclasses to interpret forward references.
        For example, List['C'] is internally stored as
        List[_ForwardRef('C')], which should evaluate to List[C],
        where C is an object found in globalns or localns (searching
        localns first, of course).
        """
        return self

    def _get_type_vars(self, tvars):
        pass

    def __repr__(self):
        qname = _trim_name(self.__qualname__)
        return '%s.%s' % (self.__module__, qname)


class _TypingBase(metaclass=TypingMeta, _root=True):
    """Internal indicator of special typing constructs."""

    __slots__ = ('__weakref__',)

    def __init__(self, *args, **kwds):
        pass

    def __new__(cls, *args, **kwds):
        """Constructor.
        This only exists to give a better error message in case
        someone tries to subclass a special typing object (not a good idea).
        """
        if (len(args) == 3 and
                isinstance(args[0], str) and
                isinstance(args[1], tuple)):
            # Close enough.
            raise TypeError("Cannot subclass %r" % cls)
        return super().__new__(cls)

    # Things that are not classes also need these.
    def _eval_type(self, globalns, localns):
        return self

    def _get_type_vars(self, tvars):
        pass

    def __repr__(self):
        cls = type(self)
        qname = _trim_name(cls.__qualname__)
        return '%s.%s' % (cls.__module__, qname)

    def __call__(self, *args, **kwds):
        raise TypeError("Cannot instantiate %r" % type(self))


class _FinalTypingBase(_TypingBase, _root=True):
    """Internal mix-in class to prevent instantiation.
    Prevents instantiation unless _root=True is given in class call.
    It is used to create pseudo-singleton instances Any, Union, Optional, etc.
    """

    __slots__ = ()

    def __new__(cls, *args, _root=False, **kwds):
        self = super().__new__(cls, *args, **kwds)
        if _root is True:
            return self
        raise TypeError("Cannot instantiate %r" % cls)

    def __reduce__(self):
        return _trim_name(type(self).__name__)


class _Literal(_FinalTypingBase, _root=True):
    """A type that can be used to indicate to type checkers that the
    corresponding value has a value literally equivalent to the
    provided parameter. For example:
        var: Literal[4] = 4
    The type checker understands that 'var' is literally equal to the
    value 4 and no other value.
    Literal[...] cannot be subclassed. There is no runtime checking
    verifying that the parameter is actually a value instead of a type.
    """

    __slots__ = ('__values__',)

    def __init__(self, values=None, **kwds):
        self.__values__ = values

    def __getitem__(self, values):
        cls = type(self)
        if self.__values__ is None:
            if not isinstance(values, tuple):
                values = (values,)
            return cls(values, _root=True)
        raise TypeError('{} cannot be further subscripted'
                        .format(cls.__name__[1:]))

    def _eval_type(self, globalns, localns):
        return self

    def __repr__(self):
        r = super().__repr__()
        if self.__values__ is not None:
            r += '[{}]'.format(', '.join(map(_type_repr, self.__values__)))
        return r

    def __hash__(self):
        return hash((type(self).__name__, self.__values__))

    def __eq__(self, other):
        if not isinstance(other, _Literal):
            return NotImplemented
        if self.__values__ is not None:
            return self.__values__ == other.__values__
        return self is other


Literal = _Literal(_root=True)


def _trim_name(nm):
    whitelist = ('_TypingBase', '_FinalTypingBase')
    if nm.startswith('_') and nm not in whitelist:
        nm = nm[1:]
    return nm


def _type_repr(obj):
    """Return the repr() of an object, special-casing types (internal helper).
    If obj is a type, we return a shorter version than the default
    type.__repr__, based on the module and qualified name, which is
    typically enough to uniquely identify a type.  For everything
    else, we fall back on repr(obj).
    """
    if isinstance(obj, type):
        if obj.__module__ == 'builtins':
            return obj.__qualname__
        return f'{obj.__module__}.{obj.__qualname__}'
    if obj is ...:
        return('...')
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    return repr(obj)
