"""Metadata reads (coverage W8) and the dtype rule of their receivers' stand-ins (soundness review S2).

A METADATA READ is a receiver factory (``method_rules.RECEIVER_FACTORIES``: ``x.new_zeros(size)``, ``new_ones``,
``new_full``) whose every operand but the receiver is a literal: its value is a function of the literals and of the
receiver's dtype and device alone, which are the program's (the converter follows the source model's dtype on its
device, and a body Dynamo captured is specialized to them). Its receiver is not part of the computation
(``batch_free``): the converted model computes the read on a zero-element stand-in of the receiver's dtype on the
program's device (``stand_in``), held in the shared state in a child named apart from every attribute of the program
(no copied value shares its path), so it moves as the state does (Swin's attention mask, ``x.new_zeros((pad_H,
pad_W))`` filled at literal indices, computed out of place by ``fx_rewrites/body_rules/local_writes``).

The stand-in follows a cast of the converted model exactly when the native receiver's dtype follows a cast of the native
model (``receivers.receiver_dtype_rule``): the receivers' computations are evaluated on meta copies of the program's
state and inputs cast to every floating dtype (a cast of the model casts its registered parameters and buffers; a
tensor constant the trace attached as a plain attribute keeps its own dtype). A receiver whose dtype is that dtype at
every one FOLLOWS (the stand-in is a floating buffer, cast with the state); one whose dtype is its traced dtype at every
one is PINNED by the program (a declared factory with a ``dtype`` argument: the stand-in is made at every call at the
traced dtype on the buffer's device, ``receiver_in_program``, so no cast changes it); any other receiver, or one the
meta evaluation cannot decide, is refused by name (``MetadataReadDtypeRefusal``). A receiver of a dtype that is not
floating keeps it under every floating cast: PINNED.
"""

from mimarsinan.torch_mapping.converter_handlers.symbolic.metadata_reads.receivers import (
    FOLLOWS,
    PINNED,
    MetadataReadDtypeRefusal,
    metadata_read,
    receiver_dtype_rule,
    receiver_in_program,
    stand_in,
)

__all__ = ["FOLLOWS", "PINNED", "MetadataReadDtypeRefusal", "metadata_read", "receiver_dtype_rule",
           "receiver_in_program", "stand_in"]
