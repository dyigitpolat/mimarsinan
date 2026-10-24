"""N2 dynamic-operand products: the lowered mapper, its bank perceptrons and the consumer-group partition."""

from mimarsinan.mapping.mappers.operand.operand_mapper import (
    OperandContractionMapper,
    operand_entry_perceptrons,
    operand_mappers,
    operand_perceptrons,
)
from mimarsinan.mapping.mappers.operand.operand_perceptron import OperandPerceptron, OperandStamp

__all__ = [
    "OperandContractionMapper",
    "OperandPerceptron",
    "OperandStamp",
    "operand_entry_perceptrons",
    "operand_mappers",
    "operand_perceptrons",
]
