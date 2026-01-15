"""
Complete mapping search module with strictly typed data structures.
Enforces architecture-aware constraints on mapping structure.
"""

import os
import subprocess
import yaml
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Literal, Union, Tuple
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Data Models - Workload
# ============================================================================

class Workload(BaseModel):
    """Complete workload specification"""
    shape: str = Field(..., description="Problem shape (e.g., Convolution)")
    dimensions: Dict[str, int] = Field(..., description="All dimension sizes (C, M, R, S, N, P, Q)")
    coefficients: Optional[Dict[str, int]] = Field(None, description="Stride/dilation coefficients")
    
    # Convenience accessors
    @property
    def C(self) -> int: return self.dimensions.get("C", 1)
    @property
    def M(self) -> int: return self.dimensions.get("M", 1)
    @property
    def R(self) -> int: return self.dimensions.get("R", 1)
    @property
    def S(self) -> int: return self.dimensions.get("S", 1)
    @property
    def N(self) -> int: return self.dimensions.get("N", 1)
    @property
    def P(self) -> int: return self.dimensions.get("P", 1)
    @property
    def Q(self) -> int: return self.dimensions.get("Q", 1)


# ============================================================================
# Data Models - Architecture
# ============================================================================

class ArchitectureAttributes(BaseModel):
    """System-level architecture attributes"""
    datawidth: int
    word_bits: int = Field(..., alias="word-bits")
    technology: str
    
    model_config = {"populate_by_name": True}


class ArchitectureComponent(BaseModel):
    """Component in architecture local array (DRAM, buffers, compute units)"""
    # Store as flexible dict since attributes vary by component type
    attributes: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArchitectureComponent':
        """Create from dict where attributes are at the top level"""
        return cls(attributes=data)


class ArchitectureSubtree(BaseModel):
    """Nested subsystem in architecture hierarchy"""
    name: Optional[str] = None
    local: Optional[List['ArchitectureComponent']] = None
    subtree: Optional[List['ArchitectureSubtree']] = None


class ArchitectureSpec(BaseModel):
    """Full architecture template specification"""
    name: str
    attributes: ArchitectureAttributes
    local: Optional[List[ArchitectureComponent]] = None
    subtree: Optional[List[ArchitectureSubtree]] = None


class ArchitectureLevel(BaseModel):
    """Single level in memory hierarchy"""
    name: str
    mesh_x: Optional[int] = None
    mesh_y: Optional[int] = None
    size: Optional[int] = None  # words
    instances: Optional[int] = None


class Architecture(BaseModel):
    """Complete hardware architecture specification"""
    name: str = Field(..., description="Architecture template name")
    mesh_x: int = Field(..., description="Mesh X dimension")
    mesh_y: int = Field(..., description="Mesh Y dimension")
    total_pes: int = Field(..., description="Total processing elements")
    levels: List[ArchitectureLevel] = Field(..., description="Memory hierarchy (0=compute, 1+=storage)")
    template: str = Field(..., description="Architecture template name (e.g., 'simba')")
    spec: ArchitectureSpec = Field(..., description="Full architecture specification")
    
    model_config = {"arbitrary_types_allowed": True}


# ============================================================================
# Data Models - Strictly Typed Mapping (Architecture-Aware)
# ============================================================================

class FactorString(BaseModel):
    """Validated factor string (e.g., 'C1 M4 R7 S7 N1 P2 Q2')"""
    factors_str: str
    
    @field_validator('factors_str')
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Ensure factors string has valid format"""
        parts = v.split()
        valid_dims = {'C', 'M', 'R', 'S', 'N', 'P', 'Q'}
        for part in parts:
            if len(part) < 2 or part[0] not in valid_dims:
                raise ValueError(f"Invalid factor: {part}")
            try:
                int(part[1:])
            except ValueError:
                raise ValueError(f"Invalid factor value: {part}")
        return v
    
    def to_dict(self) -> Dict[str, int]:
        """Parse to {dim: value} dict"""
        result = {}
        for part in self.factors_str.split():
            if len(part) >= 2:
                result[part[0]] = int(part[1:])
        return result
    
    def __str__(self) -> str:
        return self.factors_str


# ============================================================================
# Level-Specific Mapping Directives
# ============================================================================

class PEWeightRegsMapping(BaseModel):
    """PEWeightRegs level: stores Weights only, temporal tiling for P/Q"""
    # Datatype (fixed, not modifiable)
    datatype_keep: List[str] = Field(default_factory=lambda: ["Weights"], description="Must keep Weights")
    datatype_bypass: List[str] = Field(default_factory=lambda: ["Inputs", "Outputs"], description="Must bypass Inputs, Outputs")
    
    # Temporal
    temporal_factors: str = Field(..., description="Temporal factors (e.g., 'C1 M1 R1 S1 N1 P2 Q2')")
    temporal_permutation: str = Field(..., description="Loop order (e.g., 'PQCMRSN')")


class PEAccuBufferMapping(BaseModel):
    """PEAccuBuffer level: stores Outputs, spatial tiling for C"""
    # Datatype (fixed, not modifiable)
    datatype_keep: List[str] = Field(default_factory=lambda: ["Outputs"], description="Must keep Outputs")
    datatype_bypass: List[str] = Field(default_factory=lambda: ["Weights", "Inputs"], description="Must bypass Weights, Inputs")
    
    # Spatial
    spatial_factors: str = Field(..., description="Spatial factors (typically C3)")
    spatial_permutation: str = Field(..., description="Spatial order")
    spatial_split: int = Field(..., description="Split index")
    
    # Temporal
    temporal_factors: str = Field(..., description="Temporal factors (typically all 1s)")
    temporal_permutation: str = Field(..., description="Loop order")


class PEWeightBufferMapping(BaseModel):
    """PEWeightBuffer level: stores Weights, temporal tiling for R/S"""
    # Datatype (fixed, not modifiable)
    datatype_keep: List[str] = Field(default_factory=lambda: ["Weights"], description="Must keep Weights")
    datatype_bypass: List[str] = Field(default_factory=lambda: ["Inputs", "Outputs"], description="Must bypass Inputs, Outputs")
    
    # Temporal
    temporal_factors: str = Field(..., description="Temporal factors (e.g., 'C1 M1 R7 S7 N1 P1 Q1')")
    temporal_permutation: str = Field(..., description="Loop order (e.g., 'RSCMNPQ')")


class PEInputBufferMapping(BaseModel):
    """PEInputBuffer level: stores Inputs, spatial tiling for M"""
    # Datatype (fixed, not modifiable)
    datatype_keep: List[str] = Field(default_factory=lambda: ["Inputs"], description="Must keep Inputs")
    datatype_bypass: List[str] = Field(default_factory=lambda: ["Weights", "Outputs"], description="Must bypass Weights, Outputs")
    
    # Spatial
    spatial_factors: str = Field(..., description="Spatial factors (e.g., 'C1 M4 R1 S1 N1 P1 Q1')")
    spatial_permutation: str = Field(..., description="Spatial order")
    spatial_split: int = Field(..., description="Split index")
    
    # Temporal
    temporal_factors: str = Field(..., description="Temporal factors (typically all 1s)")
    temporal_permutation: str = Field(..., description="Loop order")


class GlobalBufferMapping(BaseModel):
    """GlobalBuffer level: stores Inputs/Outputs, spatial and temporal tiling"""
    # Datatype (fixed, not modifiable)
    datatype_keep: List[str] = Field(default_factory=lambda: ["Inputs", "Outputs"], description="Must keep Inputs, Outputs")
    datatype_bypass: List[str] = Field(default_factory=lambda: ["Weights"], description="Must bypass Weights")
    
    # Spatial
    spatial_factors: str = Field(..., description="Spatial factors (e.g., 'C1 M2 R1 S1 N1 P1 Q1')")
    spatial_permutation: str = Field(..., description="Spatial order")
    spatial_split: int = Field(..., description="Split index")
    
    # Temporal
    temporal_factors: str = Field(..., description="Temporal factors (e.g., 'C1 M1 R1 S1 N1 P4 Q7')")
    temporal_permutation: str = Field(..., description="Loop order")


class DRAMMapping(BaseModel):
    """DRAM level: stores all dataspaces, temporal tiling only"""
    # Datatype (fixed, not modifiable)
    datatype_keep: List[str] = Field(
        default_factory=lambda: ["Weights", "Inputs", "Outputs"], 
        description="Must keep all dataspaces"
    )
    datatype_bypass: List[str] = Field(default_factory=lambda: [], description="Cannot bypass anything")
    
    # Temporal
    temporal_factors: str = Field(..., description="Temporal factors (e.g., 'C1 M8 R1 S1 N1 P14 Q8')")
    temporal_permutation: str = Field(..., description="Loop order (e.g., 'QMPCRSN')")


# ============================================================================
# Mapping Recommendation - Simplified for Optimization
# ============================================================================

class MappingRecommendation(BaseModel):
    """
    Complete mapping recommendation for optimization.
    Includes all factors, permutations, and splits for proper constraint validation.
    """
    # PEWeightRegs (temporal only)
    pe_weight_regs_temporal_factors: str = Field(..., description="PEWeightRegs temporal factors (e.g., 'C1 M1 R1 S1 N1 P2 Q2')")
    pe_weight_regs_temporal_permutation: str = Field(..., description="PEWeightRegs temporal permutation (e.g., 'PQCMRSN')")
    
    # PEAccuBuffer (spatial + temporal)
    pe_accu_buffer_spatial_factors: str = Field(..., description="PEAccuBuffer spatial factors (e.g., 'C3 M1 R1 S1 N1 P1 Q1')")
    pe_accu_buffer_spatial_permutation: str = Field(..., description="PEAccuBuffer spatial permutation")
    pe_accu_buffer_spatial_split: int = Field(..., description="PEAccuBuffer spatial split index")
    pe_accu_buffer_temporal_factors: str = Field(..., description="PEAccuBuffer temporal factors (e.g., 'C1 M1 R1 S1 N1 P1 Q1')")
    pe_accu_buffer_temporal_permutation: str = Field(..., description="PEAccuBuffer temporal permutation")
    
    # PEWeightBuffer (temporal only)
    pe_weight_buffer_temporal_factors: str = Field(..., description="PEWeightBuffer temporal factors (e.g., 'C1 M1 R7 S7 N1 P1 Q1')")
    pe_weight_buffer_temporal_permutation: str = Field(..., description="PEWeightBuffer temporal permutation (e.g., 'RSCMNPQ')")
    
    # PEInputBuffer (spatial + temporal)
    pe_input_buffer_spatial_factors: str = Field(..., description="PEInputBuffer spatial factors (e.g., 'C1 M4 R1 S1 N1 P1 Q1')")
    pe_input_buffer_spatial_permutation: str = Field(..., description="PEInputBuffer spatial permutation")
    pe_input_buffer_spatial_split: int = Field(..., description="PEInputBuffer spatial split index")
    pe_input_buffer_temporal_factors: str = Field(..., description="PEInputBuffer temporal factors (e.g., 'C1 M1 R1 S1 N1 P1 Q1')")
    pe_input_buffer_temporal_permutation: str = Field(..., description="PEInputBuffer temporal permutation")
    
    # GlobalBuffer (spatial + temporal)
    global_buffer_spatial_factors: str = Field(..., description="GlobalBuffer spatial factors (e.g., 'C1 M2 R1 S1 N1 P1 Q1')")
    global_buffer_spatial_permutation: str = Field(..., description="GlobalBuffer spatial permutation")
    global_buffer_spatial_split: int = Field(..., description="GlobalBuffer spatial split index")
    global_buffer_temporal_factors: str = Field(..., description="GlobalBuffer temporal factors (e.g., 'C1 M1 R1 S1 N1 P4 Q7')")
    global_buffer_temporal_permutation: str = Field(..., description="GlobalBuffer temporal permutation")
    
    # DRAM (temporal only)
    dram_temporal_factors: str = Field(..., description="DRAM temporal factors (e.g., 'C1 M8 R1 S1 N1 P14 Q8')")
    dram_temporal_permutation: str = Field(..., description="DRAM temporal permutation (e.g., 'QMPCRSN')")
    
    @field_validator('pe_weight_regs_temporal_factors', 'pe_accu_buffer_spatial_factors', 'pe_accu_buffer_temporal_factors',
                     'pe_weight_buffer_temporal_factors', 'pe_input_buffer_spatial_factors', 'pe_input_buffer_temporal_factors',
                     'global_buffer_spatial_factors', 'global_buffer_temporal_factors', 'dram_temporal_factors')
    @classmethod
    def validate_temporal_factors(cls, v: str) -> str:
        """Validate temporal factors format"""
        if not v or not isinstance(v, str):
            raise ValueError(f"Temporal factors must be a non-empty string, got: {v}")
        
        # Clean up common LLM mistakes
        v = v.strip()
        
        # Check for malformed patterns like "P=2,Q=2" and convert to "P2 Q2"
        if '=' in v:
            # Convert "P=2,Q=2" to "P2 Q2"
            parts = v.replace(',', ' ').split()
            cleaned_parts = []
            for part in parts:
                if '=' in part:
                    dim, val = part.split('=', 1)
                    cleaned_parts.append(f"{dim.strip()}{val.strip()}")
                else:
                    cleaned_parts.append(part)
            v = ' '.join(cleaned_parts)
        
        # Validate format using existing FactorString logic
        parts = v.split()
        valid_dims = {'C', 'M', 'R', 'S', 'N', 'P', 'Q'}
        for part in parts:
            if len(part) < 2 or part[0] not in valid_dims:
                raise ValueError(f"Invalid temporal factor format: '{part}' in '{v}'. Expected format: 'C1 M1 R1 S1 N1 P2 Q2'")
            try:
                int(part[1:])
            except ValueError:
                raise ValueError(f"Invalid temporal factor value: '{part}' in '{v}'. Expected format: 'C1 M1 R1 S1 N1 P2 Q2'")
        
        return v
    
    def to_string(self) -> str:
        """Generate human-readable summary of all factors"""
        return (
            f"PEWeightRegs: {self.pe_weight_regs_temporal_factors} perm={self.pe_weight_regs_temporal_permutation}, "
            f"PEAccuBuffer: spatial={self.pe_accu_buffer_spatial_factors} split={self.pe_accu_buffer_spatial_split} "
            f"perm={self.pe_accu_buffer_spatial_permutation} temporal={self.pe_accu_buffer_temporal_factors} "
            f"perm={self.pe_accu_buffer_temporal_permutation}, "
            f"PEWeightBuffer: {self.pe_weight_buffer_temporal_factors} perm={self.pe_weight_buffer_temporal_permutation}, "
            f"PEInputBuffer: spatial={self.pe_input_buffer_spatial_factors} split={self.pe_input_buffer_spatial_split} "
            f"perm={self.pe_input_buffer_spatial_permutation} temporal={self.pe_input_buffer_temporal_factors} "
            f"perm={self.pe_input_buffer_temporal_permutation}, "
            f"GlobalBuffer: spatial={self.global_buffer_spatial_factors} split={self.global_buffer_spatial_split} "
            f"perm={self.global_buffer_spatial_permutation} temporal={self.global_buffer_temporal_factors} "
            f"perm={self.global_buffer_temporal_permutation}, "
            f"DRAM: {self.dram_temporal_factors} perm={self.dram_temporal_permutation}"
        )

        
def extract_recommendation_from_mapping(mapping: 'Mapping') -> 'MappingRecommendation':
    """Extract MappingRecommendation with all factors, permutations, and splits from a Mapping."""
    return MappingRecommendation(
        pe_weight_regs_temporal_factors=mapping.pe_weight_regs.temporal_factors,
        pe_weight_regs_temporal_permutation=mapping.pe_weight_regs.temporal_permutation,
        pe_accu_buffer_spatial_factors=mapping.pe_accu_buffer.spatial_factors,
        pe_accu_buffer_spatial_permutation=mapping.pe_accu_buffer.spatial_permutation,
        pe_accu_buffer_spatial_split=mapping.pe_accu_buffer.spatial_split,
        pe_accu_buffer_temporal_factors=mapping.pe_accu_buffer.temporal_factors,
        pe_accu_buffer_temporal_permutation=mapping.pe_accu_buffer.temporal_permutation,
        pe_weight_buffer_temporal_factors=mapping.pe_weight_buffer.temporal_factors,
        pe_weight_buffer_temporal_permutation=mapping.pe_weight_buffer.temporal_permutation,
        pe_input_buffer_spatial_factors=mapping.pe_input_buffer.spatial_factors,
        pe_input_buffer_spatial_permutation=mapping.pe_input_buffer.spatial_permutation,
        pe_input_buffer_spatial_split=mapping.pe_input_buffer.spatial_split,
        pe_input_buffer_temporal_factors=mapping.pe_input_buffer.temporal_factors,
        pe_input_buffer_temporal_permutation=mapping.pe_input_buffer.temporal_permutation,
        global_buffer_spatial_factors=mapping.global_buffer.spatial_factors,
        global_buffer_spatial_permutation=mapping.global_buffer.spatial_permutation,
        global_buffer_spatial_split=mapping.global_buffer.spatial_split,
        global_buffer_temporal_factors=mapping.global_buffer.temporal_factors,
        global_buffer_temporal_permutation=mapping.global_buffer.temporal_permutation,
        dram_temporal_factors=mapping.dram.temporal_factors,
        dram_temporal_permutation=mapping.dram.temporal_permutation
    )

class SimpleMappingRecommendation(BaseModel):
    """
    Simplified mapping recommendation that only tweaks resilient parameters:
    - Permutations (loop orders) at each level
    - Spatial split indices where applicable
    
    This avoids fragile factor-product constraints while still exploring
    performance-relevant scheduling decisions.
    
    CRITICAL: Permutation fields must ONLY contain dimension letters (e.g., 'PQCMRSN'),
    NOT factor strings (e.g., 'C1 M1 R1'). Split fields must be integers (0 or 1).
    """
    # PEWeightRegs (temporal only)
    pe_weight_regs_temporal_permutation: str = Field(..., description="PEWeightRegs temporal permutation - ONLY dimension letters like 'PQCMRSN', NOT factors like 'C1 M1'")
    
    # PEAccuBuffer (spatial + temporal)
    pe_accu_buffer_spatial_permutation: str = Field(..., description="PEAccuBuffer spatial permutation - ONLY dimension letters like 'CMRSNPQ'")
    pe_accu_buffer_spatial_split: int = Field(..., description="PEAccuBuffer spatial split index (0 or 1)")
    pe_accu_buffer_temporal_permutation: str = Field(..., description="PEAccuBuffer temporal permutation - ONLY dimension letters like 'CMRSNPQ'")
    
    # PEWeightBuffer (temporal only)
    pe_weight_buffer_temporal_permutation: str = Field(..., description="PEWeightBuffer temporal permutation - ONLY dimension letters like 'RSCMNPQ', NOT factors")
    
    # PEInputBuffer (spatial + temporal)
    pe_input_buffer_spatial_permutation: str = Field(..., description="PEInputBuffer spatial permutation - ONLY dimension letters like 'MCRSNPQ'")
    pe_input_buffer_spatial_split: int = Field(..., description="PEInputBuffer spatial split index (0 or 1)")
    pe_input_buffer_temporal_permutation: str = Field(..., description="PEInputBuffer temporal permutation - ONLY dimension letters like 'CMRSNPQ'")
    
    # GlobalBuffer (spatial + temporal)
    global_buffer_spatial_permutation: str = Field(..., description="GlobalBuffer spatial permutation - ONLY dimension letters like 'MCRSNPQ'")
    global_buffer_spatial_split: int = Field(..., description="GlobalBuffer spatial split index (0 or 1)")
    global_buffer_temporal_permutation: str = Field(..., description="GlobalBuffer temporal permutation - ONLY dimension letters like 'PQCMRSN'")
    
    # DRAM (temporal only)
    dram_temporal_permutation: str = Field(..., description="DRAM temporal permutation - ONLY dimension letters like 'QMPCRSN', NOT factors")
    
    def to_string(self) -> str:
        """Human-readable summary focusing on permutations and splits."""
        return (
            f"PEWeightRegs: perm={self.pe_weight_regs_temporal_permutation}, "
            f"PEAccuBuffer: split={self.pe_accu_buffer_spatial_split} "
            f"sp_perm={self.pe_accu_buffer_spatial_permutation} "
            f"t_perm={self.pe_accu_buffer_temporal_permutation}, "
            f"PEWeightBuffer: t_perm={self.pe_weight_buffer_temporal_permutation}, "
            f"PEInputBuffer: split={self.pe_input_buffer_spatial_split} "
            f"sp_perm={self.pe_input_buffer_spatial_permutation} "
            f"t_perm={self.pe_input_buffer_temporal_permutation}, "
            f"GlobalBuffer: split={self.global_buffer_spatial_split} "
            f"sp_perm={self.global_buffer_spatial_permutation} "
            f"t_perm={self.global_buffer_temporal_permutation}, "
            f"DRAM: t_perm={self.dram_temporal_permutation}"
        )

def recommendation_to_mapping(
    recommendation: 'MappingRecommendation',
    workload: 'Workload',
    architecture: 'Architecture'
) -> 'Mapping':
    """
    Convert a MappingRecommendation to a full Mapping object.
    Adds fixed datatype bindings for each hierarchy level.
    """
    return Mapping(
        pe_weight_regs=PEWeightRegsMapping(
            datatype_keep=["Weights"],
            datatype_bypass=["Inputs", "Outputs"],
            temporal_factors=recommendation.pe_weight_regs_temporal_factors,
            temporal_permutation=recommendation.pe_weight_regs_temporal_permutation
        ),
        pe_accu_buffer=PEAccuBufferMapping(
            datatype_keep=["Outputs"],
            datatype_bypass=["Weights", "Inputs"],
            spatial_factors=recommendation.pe_accu_buffer_spatial_factors,
            spatial_permutation=recommendation.pe_accu_buffer_spatial_permutation,
            spatial_split=recommendation.pe_accu_buffer_spatial_split,
            temporal_factors=recommendation.pe_accu_buffer_temporal_factors,
            temporal_permutation=recommendation.pe_accu_buffer_temporal_permutation
        ),
        pe_weight_buffer=PEWeightBufferMapping(
            datatype_keep=["Weights"],
            datatype_bypass=["Inputs", "Outputs"],
            temporal_factors=recommendation.pe_weight_buffer_temporal_factors,
            temporal_permutation=recommendation.pe_weight_buffer_temporal_permutation
        ),
        pe_input_buffer=PEInputBufferMapping(
            datatype_keep=["Inputs"],
            datatype_bypass=["Weights", "Outputs"],
            spatial_factors=recommendation.pe_input_buffer_spatial_factors,
            spatial_permutation=recommendation.pe_input_buffer_spatial_permutation,
            spatial_split=recommendation.pe_input_buffer_spatial_split,
            temporal_factors=recommendation.pe_input_buffer_temporal_factors,
            temporal_permutation=recommendation.pe_input_buffer_temporal_permutation
        ),
        global_buffer=GlobalBufferMapping(
            datatype_keep=["Inputs", "Outputs"],
            datatype_bypass=["Weights"],
            spatial_factors=recommendation.global_buffer_spatial_factors,
            spatial_permutation=recommendation.global_buffer_spatial_permutation,
            spatial_split=recommendation.global_buffer_spatial_split,
            temporal_factors=recommendation.global_buffer_temporal_factors,
            temporal_permutation=recommendation.global_buffer_temporal_permutation
        ),
        dram=DRAMMapping(
            datatype_keep=["Weights", "Inputs", "Outputs"],
            datatype_bypass=[],
            temporal_factors=recommendation.dram_temporal_factors,
            temporal_permutation=recommendation.dram_temporal_permutation
        )
    )

def merge_simple_recommendation_with_mapping(
    recommendation: SimpleMappingRecommendation,
    base_mapping: 'Mapping'
) -> 'Mapping':
    """
    Merge a SimpleMappingRecommendation into a base Mapping, updating only
    permutations and split indices. All factor strings are preserved from
    the base mapping to maintain factor-product constraints.
    """
    pe_weight_regs = PEWeightRegsMapping(
        temporal_factors=base_mapping.pe_weight_regs.temporal_factors,
        temporal_permutation=recommendation.pe_weight_regs_temporal_permutation,
    )
    
    pe_accu_buffer = PEAccuBufferMapping(
        spatial_factors=base_mapping.pe_accu_buffer.spatial_factors,
        spatial_permutation=recommendation.pe_accu_buffer_spatial_permutation,
        spatial_split=recommendation.pe_accu_buffer_spatial_split,
        temporal_factors=base_mapping.pe_accu_buffer.temporal_factors,
        temporal_permutation=recommendation.pe_accu_buffer_temporal_permutation,
    )
    
    pe_weight_buffer = PEWeightBufferMapping(
        temporal_factors=base_mapping.pe_weight_buffer.temporal_factors,
        temporal_permutation=recommendation.pe_weight_buffer_temporal_permutation,
    )
    
    pe_input_buffer = PEInputBufferMapping(
        spatial_factors=base_mapping.pe_input_buffer.spatial_factors,
        spatial_permutation=recommendation.pe_input_buffer_spatial_permutation,
        spatial_split=recommendation.pe_input_buffer_spatial_split,
        temporal_factors=base_mapping.pe_input_buffer.temporal_factors,
        temporal_permutation=recommendation.pe_input_buffer_temporal_permutation,
    )
    
    global_buffer = GlobalBufferMapping(
        spatial_factors=base_mapping.global_buffer.spatial_factors,
        spatial_permutation=recommendation.global_buffer_spatial_permutation,
        spatial_split=recommendation.global_buffer_spatial_split,
        temporal_factors=base_mapping.global_buffer.temporal_factors,
        temporal_permutation=recommendation.global_buffer_temporal_permutation,
    )
    
    dram = DRAMMapping(
        temporal_factors=base_mapping.dram.temporal_factors,
        temporal_permutation=recommendation.dram_temporal_permutation,
    )
    
    return Mapping(
        pe_weight_regs=pe_weight_regs,
        pe_accu_buffer=pe_accu_buffer,
        pe_weight_buffer=pe_weight_buffer,
        pe_input_buffer=pe_input_buffer,
        global_buffer=global_buffer,
        dram=dram,
    )

def merge_recommendation_with_mapping(
    recommendation: MappingRecommendation,
    base_mapping: 'Mapping'
) -> 'Mapping':
    """
    Merge MappingRecommendation with all factors, permutations, and splits into base Mapping structure.
    Returns new Mapping with updated parameters from recommendation.
    """
    # Create new level objects with updated parameters from recommendation
    pe_weight_regs = PEWeightRegsMapping(
        temporal_factors=recommendation.pe_weight_regs_temporal_factors,
        temporal_permutation=recommendation.pe_weight_regs_temporal_permutation
    )
    
    pe_accu_buffer = PEAccuBufferMapping(
        spatial_factors=recommendation.pe_accu_buffer_spatial_factors,
        spatial_permutation=recommendation.pe_accu_buffer_spatial_permutation,
        spatial_split=recommendation.pe_accu_buffer_spatial_split,
        temporal_factors=recommendation.pe_accu_buffer_temporal_factors,
        temporal_permutation=recommendation.pe_accu_buffer_temporal_permutation
    )
    
    pe_weight_buffer = PEWeightBufferMapping(
        temporal_factors=recommendation.pe_weight_buffer_temporal_factors,
        temporal_permutation=recommendation.pe_weight_buffer_temporal_permutation
    )
    
    pe_input_buffer = PEInputBufferMapping(
        spatial_factors=recommendation.pe_input_buffer_spatial_factors,
        spatial_permutation=recommendation.pe_input_buffer_spatial_permutation,
        spatial_split=recommendation.pe_input_buffer_spatial_split,
        temporal_factors=recommendation.pe_input_buffer_temporal_factors,
        temporal_permutation=recommendation.pe_input_buffer_temporal_permutation
    )
    
    global_buffer = GlobalBufferMapping(
        spatial_factors=recommendation.global_buffer_spatial_factors,
        spatial_permutation=recommendation.global_buffer_spatial_permutation,
        spatial_split=recommendation.global_buffer_spatial_split,
        temporal_factors=recommendation.global_buffer_temporal_factors,
        temporal_permutation=recommendation.global_buffer_temporal_permutation
    )
    
    dram = DRAMMapping(
        temporal_factors=recommendation.dram_temporal_factors,
        temporal_permutation=recommendation.dram_temporal_permutation
    )
    
    return Mapping(
        pe_weight_regs=pe_weight_regs,
        pe_accu_buffer=pe_accu_buffer,
        pe_weight_buffer=pe_weight_buffer,
        pe_input_buffer=pe_input_buffer,
        global_buffer=global_buffer,
        dram=dram
    )


# ============================================================================
# Complete Mapping - Strict Hierarchical Structure
# ============================================================================

class Mapping(BaseModel):
    """
    Complete dataflow mapping with strict architecture-aware structure.
    Enforces exact hierarchy: PEWeightRegs -> PEAccuBuffer -> PEWeightBuffer 
    -> PEInputBuffer -> GlobalBuffer -> DRAM
    """
    pe_weight_regs: PEWeightRegsMapping = Field(..., description="PEWeightRegs level (Weights, temporal P/Q)")
    pe_accu_buffer: PEAccuBufferMapping = Field(..., description="PEAccuBuffer level (Outputs, spatial C)")
    pe_weight_buffer: PEWeightBufferMapping = Field(..., description="PEWeightBuffer level (Weights, temporal R/S)")
    pe_input_buffer: PEInputBufferMapping = Field(..., description="PEInputBuffer level (Inputs, spatial M)")
    global_buffer: GlobalBufferMapping = Field(..., description="GlobalBuffer level (Inputs/Outputs, spatial M, temporal P/Q)")
    dram: DRAMMapping = Field(..., description="DRAM level (all dataspaces, temporal M/P/Q)")
    
    def to_yaml_directives(self) -> List[Dict[str, Any]]:
        """Convert to YAML mapping directive list (order matters!)"""
        directives = []
        
        # Datatype directives (all levels first)
        directives.append({
            "target": "PEWeightRegs",
            "type": "datatype",
            "keep": list(self.pe_weight_regs.datatype_keep),
            "bypass": list(self.pe_weight_regs.datatype_bypass)
        })
        directives.append({
            "target": "PEAccuBuffer",
            "type": "datatype",
            "keep": list(self.pe_accu_buffer.datatype_keep),
            "bypass": list(self.pe_accu_buffer.datatype_bypass)
        })
        directives.append({
            "target": "PEWeightBuffer",
            "type": "datatype",
            "keep": list(self.pe_weight_buffer.datatype_keep),
            "bypass": list(self.pe_weight_buffer.datatype_bypass)
        })
        directives.append({
            "target": "PEInputBuffer",
            "type": "datatype",
            "keep": list(self.pe_input_buffer.datatype_keep),
            "bypass": list(self.pe_input_buffer.datatype_bypass)
        })
        directives.append({
            "target": "GlobalBuffer",
            "type": "datatype",
            "keep": list(self.global_buffer.datatype_keep),
            "bypass": list(self.global_buffer.datatype_bypass)
        })
        directives.append({
            "target": "DRAM",
            "type": "datatype",
            "keep": list(self.dram.datatype_keep),
            "bypass": list(self.dram.datatype_bypass)
        })
        
        # Spatial/Temporal directives (interleaved per level)
        directives.append({
            "target": "PEWeightRegs",
            "type": "temporal",
            "factors": self.pe_weight_regs.temporal_factors,
            "permutation": self.pe_weight_regs.temporal_permutation
        })
        
        directives.append({
            "target": "PEAccuBuffer",
            "type": "spatial",
            "factors": self.pe_accu_buffer.spatial_factors,
            "permutation": self.pe_accu_buffer.spatial_permutation,
            "split": self.pe_accu_buffer.spatial_split
        })
        directives.append({
            "target": "PEAccuBuffer",
            "type": "temporal",
            "factors": self.pe_accu_buffer.temporal_factors,
            "permutation": self.pe_accu_buffer.temporal_permutation
        })
        
        directives.append({
            "target": "PEWeightBuffer",
            "type": "temporal",
            "factors": self.pe_weight_buffer.temporal_factors,
            "permutation": self.pe_weight_buffer.temporal_permutation
        })
        
        directives.append({
            "target": "PEInputBuffer",
            "type": "spatial",
            "factors": self.pe_input_buffer.spatial_factors,
            "permutation": self.pe_input_buffer.spatial_permutation,
            "split": self.pe_input_buffer.spatial_split
        })
        directives.append({
            "target": "PEInputBuffer",
            "type": "temporal",
            "factors": self.pe_input_buffer.temporal_factors,
            "permutation": self.pe_input_buffer.temporal_permutation
        })
        
        directives.append({
            "target": "GlobalBuffer",
            "type": "spatial",
            "factors": self.global_buffer.spatial_factors,
            "permutation": self.global_buffer.spatial_permutation,
            "split": self.global_buffer.spatial_split
        })
        directives.append({
            "target": "GlobalBuffer",
            "type": "temporal",
            "factors": self.global_buffer.temporal_factors,
            "permutation": self.global_buffer.temporal_permutation
        })
        
        directives.append({
            "target": "DRAM",
            "type": "temporal",
            "factors": self.dram.temporal_factors,
            "permutation": self.dram.temporal_permutation
        })
        
        return directives
    
    def get_factors_dict(self) -> Dict[str, Dict[str, int]]:
        """Extract all factors as nested dict: {level_type: {dim: value}}"""
        result = {}
        
        # Parse factor strings
        for level_name, level_obj in [
            ("PEWeightRegs_temporal", self.pe_weight_regs),
            ("PEAccuBuffer_spatial", self.pe_accu_buffer),
            ("PEAccuBuffer_temporal", self.pe_accu_buffer),
            ("PEWeightBuffer_temporal", self.pe_weight_buffer),
            ("PEInputBuffer_spatial", self.pe_input_buffer),
            ("PEInputBuffer_temporal", self.pe_input_buffer),
            ("GlobalBuffer_spatial", self.global_buffer),
            ("GlobalBuffer_temporal", self.global_buffer),
            ("DRAM_temporal", self.dram),
        ]:
            if hasattr(level_obj, 'spatial_factors') and '_spatial' in level_name:
                factors = {}
                for part in level_obj.spatial_factors.split():
                    if len(part) >= 2:
                        factors[part[0]] = int(part[1:])
                result[level_name] = factors
            elif hasattr(level_obj, 'temporal_factors') and '_temporal' in level_name:
                factors = {}
                for part in level_obj.temporal_factors.split():
                    if len(part) >= 2:
                        factors[part[0]] = int(part[1:])
                result[level_name] = factors
        
        return result
    
    def to_string(self) -> str:
        """Human-readable summary of key factors"""
        fd = self.get_factors_dict()
        
        pe_a_s = fd.get("PEAccuBuffer_spatial", {})
        pe_ib_s = fd.get("PEInputBuffer_spatial", {})
        gb_s = fd.get("GlobalBuffer_spatial", {})
        
        pe_w_t = fd.get("PEWeightRegs_temporal", {})
        pe_wb_t = fd.get("PEWeightBuffer_temporal", {})
        gb_t = fd.get("GlobalBuffer_temporal", {})
        dram_t = fd.get("DRAM_temporal", {})
        
        return (
            f"C{pe_a_s.get('C', 1)} "
            f"M{pe_ib_s.get('M', 1)}×{gb_s.get('M', 1)}×{dram_t.get('M', 1)} "
            f"R{pe_wb_t.get('R', 1)} S{pe_wb_t.get('S', 1)} "
            f"P{pe_w_t.get('P', 1)}×{gb_t.get('P', 1)}×{dram_t.get('P', 1)} "
            f"Q{pe_w_t.get('Q', 1)}×{gb_t.get('Q', 1)}×{dram_t.get('Q', 1)}"
        )


class Metrics(BaseModel):
    """Evaluation metrics"""
    energy: float = Field(..., description="Energy in pJ")
    cycles: int = Field(..., description="Execution cycles")
    area: float = Field(..., description="Area in um²")
    
    def to_string(self) -> str:
        return f"E={self.energy:.2f}pJ C={self.cycles} A={self.area:.2f}um²"


class EvaluationError(BaseModel):
    """Evaluation error with detailed information"""
    message: str = Field(..., description="Error message")
    details: List[str] = Field(default_factory=list, description="Detailed error messages")
    stdout: Optional[str] = Field(None, description="Raw stdout from moham")
    stderr: Optional[str] = Field(None, description="Raw stderr from moham")
    return_code: Optional[int] = Field(None, description="Process return code")
    
    def to_string(self) -> str:
        lines = [f"Error: {self.message}"]
        if self.details:
            lines.append("Details:")
            for detail in self.details:
                lines.append(f"  - {detail}")
        if self.return_code is not None:
            lines.append(f"Return code: {self.return_code}")
        return "\n".join(lines)


class EvaluationResult(BaseModel):
    """Result of mapping evaluation - either success or failure"""
    success: bool = Field(..., description="Whether evaluation succeeded")
    metrics: Optional[Metrics] = Field(None, description="Metrics if successful")
    error: Optional[EvaluationError] = Field(None, description="Error if failed")
    
    def __str__(self) -> str:
        if self.success and self.metrics:
            return f"Success: {self.metrics.to_string()}"
        elif self.error:
            return f"Failed: {self.error.to_string()}"
        else:
            return "Unknown result"


# Union type for evaluation results
EvaluationOutcome = Union[Metrics, EvaluationError]


class MappingResult(BaseModel):
    """Result of mapping evaluation with recommendation and insights"""
    recommendation: Union[MappingRecommendation, 'SimpleMappingRecommendation'] = Field(..., description="The mapping recommendation")
    result: EvaluationOutcome = Field(..., description="Evaluation result (success or failure)")
    insight: str = Field(default="", description="Insight about the mapping")


# ============================================================================
# Bundle Parsing
# ============================================================================

def parse_bundle(bundle_path: str) -> Tuple[Workload, Architecture, Mapping]:
    """
    Parse complete YAML bundle into strictly typed objects.
    
    Returns:
        (workload, architecture, mapping)
    """

    print(f"Parsing bundle file {bundle_path}")

    with open(bundle_path, 'r') as f:
        bundle = yaml.safe_load(f)

    if bundle is None:
        raise ValueError(f"Failed to load bundle file {bundle_path}")
    
    # ===== Parse Workload =====
    prob = bundle["workload"]["problem"]
    inst = prob["instance"]

    print(f"Workload and instance parsed successfully.")
    
    dimensions = {k: v for k, v in inst.items() if k in ["C", "M", "R", "S", "N", "P", "Q"]}
    coeffs = {k: v for k, v in inst.items() if k not in dimensions}
    
    workload = Workload(
        shape=prob["shape"]["name"],
        dimensions=dimensions,
        coefficients=coeffs if coeffs else None
    )
    
    # ===== Parse Architecture =====
    # Note: arch_min (if present) is ignored. Minimal architectures are now
    # computed at runtime by the evaluator from the template architecture.
    arch_dict = bundle.get("architecture", {})
    
    levels: List[ArchitectureLevel] = []
    mesh_x = 1
    mesh_y = 1
    arch_name = arch_dict.get("name", "Unknown")
    
    # Build a minimal logical hierarchy for documentation/constraints. The
    # actual hardware topology comes from the Timeloop/Simba templates.
    level_names = ["LMAC", "PEWeightRegs", "PEAccuBuffer",
                   "PEWeightBuffer", "PEInputBuffer", "GlobalBuffer"]
    for name in level_names:
        levels.append(ArchitectureLevel(name=name))

    print(f"Architecture levels parsed successfully.")
    
    # Parse architecture dict into structured ArchitectureSpec (informational)
    attributes = ArchitectureAttributes(
        datawidth=arch_dict["attributes"]["datawidth"],
        word_bits=arch_dict["attributes"]["word-bits"],
        technology=arch_dict["attributes"]["technology"]
    )
    
    # Parse local components
    local_components = None
    if "local" in arch_dict:
        local_components = [ArchitectureComponent.from_dict(comp) for comp in arch_dict["local"]]
    
    # Parse subtree (recursive if needed, or store as dict for simplicity)
    subtree = None
    if "subtree" in arch_dict:
        subtree = []
        for sub in arch_dict["subtree"]:
            # Parse local components in subtree
            local_in_subtree = None
            if "local" in sub:
                local_in_subtree = [ArchitectureComponent.from_dict(comp) for comp in sub["local"]]
            
            # Recursively parse nested subtrees
            nested_subtree = None
            if "subtree" in sub:
                nested_subtree = [ArchitectureSubtree(**nested_sub) for nested_sub in sub["subtree"]]
            
            subtree.append(ArchitectureSubtree(
                name=sub.get("name"),
                local=local_in_subtree,
                subtree=nested_subtree
            ))
    
    print(f"Subtree parsed successfully.")
    
    spec = ArchitectureSpec(
        name=arch_dict["name"],
        attributes=attributes,
        local=local_components,
        subtree=subtree
    )
    
    template = bundle.get("meta", {}).get("template", "simba")
    
    architecture = Architecture(
        name=arch_name,
        mesh_x=mesh_x,
        mesh_y=mesh_y,
        total_pes=mesh_x * mesh_y,
        levels=levels,
        template=template,
        spec=spec
    )
    
    # ===== Parse Mapping - Strictly Typed by Level =====
    mapping_list = bundle["mapping"]

    print(f"Mapping list parsed successfully.")
    
    # Extract directives by target and type
    def find_directive(target: str, dtype: str) -> Optional[Dict]:
        for d in mapping_list:
            if d.get("target") == target and d.get("type") == dtype:
                return d

        print(f"Directive not found for target {target} and type {dtype}")
        return None

    # Canonical identity factors/permutation used when a directive is absent.
    # This represents "no tiling" at that level and is semantically neutral.
    IDENTITY_FACTORS = "C1 M1 R1 S1 N1 P1 Q1"
    IDENTITY_PERM = "CMRSNPQ"

    def default_spatial_dict() -> Dict[str, Any]:
        return {
            "factors": IDENTITY_FACTORS,
            "permutation": IDENTITY_PERM,
            "split": 0,
        }

    def default_temporal_dict() -> Dict[str, Any]:
        return {
            "factors": IDENTITY_FACTORS,
            "permutation": IDENTITY_PERM,
        }

    # ----- PEWeightRegs (temporal only) -----
    pe_wr_t = find_directive("PEWeightRegs", "temporal")
    if pe_wr_t is None:
        raise ValueError(
            "Bundle mapping is missing required temporal directive for PEWeightRegs. "
            "This is inconsistent with Simba constraints."
        )
    pe_weight_regs = PEWeightRegsMapping(
        temporal_factors=pe_wr_t["factors"],
        temporal_permutation=pe_wr_t["permutation"],
    )

    # ----- PEAccuBuffer (spatial + temporal) -----
    pe_ab_s = find_directive("PEAccuBuffer", "spatial")
    if pe_ab_s is None:
        # Treat absence of spatial directive as identity spatial tiling.
        pe_ab_s = default_spatial_dict()
    pe_ab_t = find_directive("PEAccuBuffer", "temporal")
    if pe_ab_t is None:
        pe_ab_t = default_temporal_dict()
    pe_accu_buffer = PEAccuBufferMapping(
        spatial_factors=pe_ab_s["factors"],
        spatial_permutation=pe_ab_s["permutation"],
        spatial_split=pe_ab_s["split"],
        temporal_factors=pe_ab_t["factors"],
        temporal_permutation=pe_ab_t["permutation"],
    )

    # ----- PEWeightBuffer (temporal only) -----
    pe_wb_t = find_directive("PEWeightBuffer", "temporal")
    if pe_wb_t is None:
        raise ValueError(
            "Bundle mapping is missing required temporal directive for PEWeightBuffer."
        )
    pe_weight_buffer = PEWeightBufferMapping(
        temporal_factors=pe_wb_t["factors"],
        temporal_permutation=pe_wb_t["permutation"],
    )

    # ----- PEInputBuffer (spatial + temporal) -----
    pe_ib_s = find_directive("PEInputBuffer", "spatial")
    if pe_ib_s is None:
        # Some Medea bundles omit spatial directives for PEInputBuffer when
        # there is effectively no spatial tiling. Use identity spatial tiling.
        pe_ib_s = default_spatial_dict()
    pe_ib_t = find_directive("PEInputBuffer", "temporal")
    if pe_ib_t is None:
        pe_ib_t = default_temporal_dict()
    pe_input_buffer = PEInputBufferMapping(
        spatial_factors=pe_ib_s["factors"],
        spatial_permutation=pe_ib_s["permutation"],
        spatial_split=pe_ib_s["split"],
        temporal_factors=pe_ib_t["factors"],
        temporal_permutation=pe_ib_t["permutation"],
    )

    # ----- GlobalBuffer (spatial + temporal) -----
    gb_s = find_directive("GlobalBuffer", "spatial")
    if gb_s is None:
        # Some bundles omit explicit spatial directives for GlobalBuffer.
        gb_s = default_spatial_dict()
    gb_t = find_directive("GlobalBuffer", "temporal")
    if gb_t is None:
        gb_t = default_temporal_dict()
    global_buffer = GlobalBufferMapping(
        spatial_factors=gb_s["factors"],
        spatial_permutation=gb_s["permutation"],
        spatial_split=gb_s["split"],
        temporal_factors=gb_t["factors"],
        temporal_permutation=gb_t["permutation"],
    )

    # ----- DRAM (temporal only) -----
    dram_t = find_directive("DRAM", "temporal")
    if dram_t is None:
        raise ValueError(
            "Bundle mapping is missing required temporal directive for DRAM."
        )
    dram = DRAMMapping(
        temporal_factors=dram_t["factors"],
        temporal_permutation=dram_t["permutation"],
    )

    mapping = Mapping(
        pe_weight_regs=pe_weight_regs,
        pe_accu_buffer=pe_accu_buffer,
        pe_weight_buffer=pe_weight_buffer,
        pe_input_buffer=pe_input_buffer,
        global_buffer=global_buffer,
        dram=dram
    )
    print("Bundle parsed successfully.")
    
    return workload, architecture, mapping


# ============================================================================
# Evaluation
# ============================================================================

def build_bundle(workload: Workload, architecture: Architecture, mapping: Mapping) -> Dict:
    """
    Construct complete bundle dict from structured objects.
    Creates a minimal but valid Timeloop bundle for evaluation.
    """
    # Build workload section
    workload_dict = {
        "problem": {
            "shape": {
                "name": workload.shape,
                "dimensions": ["C", "M", "R", "S", "N", "P", "Q"],
                "coefficients": [
                    {"name": "Wstride", "default": 1},
                    {"name": "Hstride", "default": 1},
                    {"name": "Wdilation", "default": 1},
                    {"name": "Hdilation", "default": 1}
                ],
                "data-spaces": [
                    {
                        "name": "Weights",
                        "projection": [
                            [["C"]], [["M"]], [["R"]], [["S"]]
                        ]
                    },
                    {
                        "name": "Inputs",
                        "projection": [
                            [["N"]], 
                            [["C"]], 
                            [["R", "Wdilation"], ["P", "Wstride"]], 
                            [["S", "Hdilation"], ["Q", "Hstride"]]
                        ]
                    },
                    {
                        "name": "Outputs",
                        "projection": [
                            [["N"]], [["M"]], [["Q"]], [["P"]]
                        ],
                        "read-write": True
                    }
                ]
            },
            "instance": workload.dimensions.copy()
        }
    }
    
    # Add coefficients to instance
    if workload.coefficients:
        workload_dict["problem"]["instance"].update(workload.coefficients)
    
    # Build architecture dict from structured spec
    architecture_dict = {
        "name": architecture.spec.name,
        "attributes": {
            "datawidth": architecture.spec.attributes.datawidth,
            "word-bits": architecture.spec.attributes.word_bits,
            "technology": architecture.spec.attributes.technology
        }
    }
    
    if architecture.spec.local:
        architecture_dict["local"] = [comp.attributes for comp in architecture.spec.local]
    
    if architecture.spec.subtree:
        architecture_dict["subtree"] = [sub.model_dump(exclude_none=True) for sub in architecture.spec.subtree]
    
    # Build complete bundle (arch_min is intentionally omitted; it is computed
    # dynamically by the evaluator from engine specs)
    template_name = architecture.template
    bundle = {
        "meta": {
            "version": 1,
            "template": template_name,
            "workload_id": 0,
            "update_ert": True  # Required for accurate energy
        },
        "workload": workload_dict,
        "architecture": architecture_dict,
        "mapping": mapping.to_yaml_directives()
    }
    
    return bundle


def evaluate_mapping(
    workload: Workload,
    architecture: Architecture,
    mapping: Mapping,
    moham_bin: str = "/workspaces/moham/build/moham",
    timeout: int = 120
) -> EvaluationOutcome:
    """
    Evaluate complete mapping using moham --eval-bundle.
    
    Args:
        workload: Workload specification
        architecture: Architecture specification
        mapping: Complete mapping with all 6 levels
        moham_bin: Path to moham executable
        timeout: Evaluation timeout in seconds
    
    Returns:
        Metrics object if successful, EvaluationError if failed
    """
    # Build bundle from structured objects
    bundle = build_bundle(workload, architecture, mapping)
    bundle_yaml = yaml.safe_dump(bundle, sort_keys=False)

    trace_eval = bool(os.getenv("KEDI_EVAL_TRACE") or os.getenv("KEDI_LLM_TRACE"))
    if trace_eval:
        print("\n[KEDI_EVAL] Evaluating mapping")
        try:
            from mapping_core import prettify_mapping  # type: ignore[import]
        except Exception:
            prettify_mapping = None  # type: ignore[assignment]
        if prettify_mapping is not None:
            try:
                print("[KEDI_EVAL] Mapping (prettified):")
                print(prettify_mapping(mapping))
            except Exception:
                pass
        print("[KEDI_EVAL] Bundle YAML (truncated):")
        print(bundle_yaml[:4000])
    
    # Strip venv from environment (moham needs system Python with accelergy)
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONHOME")}
    if "PATH" in env:
        env["PATH"] = ":".join([p for p in env["PATH"].split(":") if "env312" not in p])
    
    # Run evaluation
    try:
        proc = subprocess.run(
            f"{moham_bin} --eval-bundle",
            input=bundle_yaml,
            text=True,
            shell=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return EvaluationError(
            message="Evaluation timeout",
            details=[f"Process timed out after {timeout} seconds"],
            return_code=-1
        )
    except Exception as e:
        return EvaluationError(
            message="Evaluation failed to start",
            details=[str(e)],
            return_code=-1
        )
    
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()

    if trace_eval:
        print("[KEDI_EVAL] moham stdout (truncated):")
        print(stdout[:2000])
        if stderr:
            print("[KEDI_EVAL] moham stderr (truncated):")
            print(stderr[:1000])
        print(f"[KEDI_EVAL] moham return_code={proc.returncode}")
    
    # Parse output
    try:
        doc = yaml.safe_load(stdout) if stdout else None
    except Exception as e:
        return EvaluationError(
            message="YAML parse failed",
            details=[str(e)],
            stdout=stdout[:500] if stdout else None,
            stderr=stderr[:500] if stderr else None,
            return_code=proc.returncode
        )
    
    # Check for errors (moham emits error YAML on failure)
    if isinstance(doc, dict) and "error" in doc:
        error_info = doc["error"]
        message = error_info.get("message", "Evaluation failed")
        details = error_info.get("details", [])
        if isinstance(details, list):
            details = [str(d) for d in details]
        else:
            details = [str(details)]
        
        return EvaluationError(
            message=message,
            details=details,
            stdout=stdout if stdout else None,
            stderr=stderr if stderr else None,
            return_code=proc.returncode
        )
    
    # Extract metrics
    if isinstance(doc, dict) and all(k in doc for k in ("energy", "cycles", "area")):
        try:
            metrics = Metrics(
                energy=float(doc["energy"]),
                cycles=int(doc["cycles"]),
                area=float(doc["area"])
            )
            return metrics
        except Exception as e:
            return EvaluationError(
                message="Malformed metrics",
                details=[str(e)],
                stdout=stdout if stdout else None,
                stderr=stderr if stderr else None,
                return_code=proc.returncode
            )
    
    # Unexpected output format
    return EvaluationError(
        message="Unexpected output format",
        details=[
            f"Expected metrics dict with keys: energy, cycles, area",
            f"Got: {type(doc)} with keys: {list(doc.keys()) if isinstance(doc, dict) else 'N/A'}"
        ],
        stdout=stdout if stdout else None,
        stderr=stderr if stderr else None,
        return_code=proc.returncode
    )


# ============================================================================
# Context Prettifiers
# ============================================================================

def prettify_workload(workload: Workload) -> str:
    """Generate human-readable workload description"""
    lines = [
        "WORKLOAD",
        "=" * 60,
        f"Shape: {workload.shape}",
        f"Dimensions:",
    ]
    
    dim_names = {
        "C": "input channels",
        "M": "output channels",
        "R": "filter height",
        "S": "filter width",
        "N": "batch size",
        "P": "output height",
        "Q": "output width"
    }
    
    for dim in ["C", "M", "R", "S", "N", "P", "Q"]:
        if dim in workload.dimensions:
            name = dim_names.get(dim, dim)
            lines.append(f"  {dim} ({name:20s}): {workload.dimensions[dim]}")
    
    if workload.coefficients:
        lines.append("Coefficients:")
        for k, v in workload.coefficients.items():
            lines.append(f"  {k:20s}: {v}")
    
    return "\n".join(lines)


def prettify_architecture(architecture: Architecture) -> str:
    """Generate human-readable architecture description"""
    lines = [
        "ARCHITECTURE",
        "=" * 60,
        f"Template: {architecture.template}",
        f"Name: {architecture.spec.name}",
        "",
        "System Attributes:",
        f"  Datawidth:  {architecture.spec.attributes.datawidth} bits",
        f"  Word-bits:  {architecture.spec.attributes.word_bits} bits",
        f"  Technology: {architecture.spec.attributes.technology}",
        "",
        # NOTE: The effective PE mesh and per-level capacities (arch_min) are
        # computed by the C++ evaluator at runtime from the template
        # architecture and current mapping. We intentionally do not try to
        # reconstruct those dynamic values here to avoid misleading the LLM.
        "Memory Hierarchy (logical levels; runtime arch_min is computed by the evaluator):",
    ]
    
    # Only print level names here; mesh/size are determined from the template
    # and arch_min inside Medea/Timeloop, not in Python.
    for i, level in enumerate(architecture.levels):
        lines.append(f"  {i}. {level.name}")
    
    # Optionally show full hierarchy
    if architecture.spec.local:
        lines.append("")
        lines.append(f"Top-level Components: {len(architecture.spec.local)}")
        
    if architecture.spec.subtree:
        lines.append(f"Subsystems: {len(architecture.spec.subtree)}")
    
    return "\n".join(lines)


def prettify_mapping(mapping: Mapping) -> str:
    """Generate human-readable mapping description"""
    lines = [
        "MAPPING",
        "=" * 60,
        f"Summary: {mapping.to_string()}",
        "",
        "Hierarchical Structure: 6 levels with fixed datatype bindings",
    ]
    
    lines.append("")
    lines.append("Complete Specification:")
    
    lines.append("\n  PEWeightRegs (Weights):")
    lines.append(f"    temporal: {mapping.pe_weight_regs.temporal_factors}")
    lines.append(f"    perm:     {mapping.pe_weight_regs.temporal_permutation}")
    
    lines.append("\n  PEAccuBuffer (Outputs):")
    lines.append(f"    spatial:  {mapping.pe_accu_buffer.spatial_factors} split={mapping.pe_accu_buffer.spatial_split}")
    lines.append(f"    perm:     {mapping.pe_accu_buffer.spatial_permutation}")
    lines.append(f"    temporal: {mapping.pe_accu_buffer.temporal_factors}")
    lines.append(f"    perm:     {mapping.pe_accu_buffer.temporal_permutation}")
    
    lines.append("\n  PEWeightBuffer (Weights):")
    lines.append(f"    temporal: {mapping.pe_weight_buffer.temporal_factors}")
    lines.append(f"    perm:     {mapping.pe_weight_buffer.temporal_permutation}")
    
    lines.append("\n  PEInputBuffer (Inputs):")
    lines.append(f"    spatial:  {mapping.pe_input_buffer.spatial_factors} split={mapping.pe_input_buffer.spatial_split}")
    lines.append(f"    perm:     {mapping.pe_input_buffer.spatial_permutation}")
    lines.append(f"    temporal: {mapping.pe_input_buffer.temporal_factors}")
    lines.append(f"    perm:     {mapping.pe_input_buffer.temporal_permutation}")
    
    lines.append("\n  GlobalBuffer (Inputs, Outputs):")
    lines.append(f"    spatial:  {mapping.global_buffer.spatial_factors} split={mapping.global_buffer.spatial_split}")
    lines.append(f"    perm:     {mapping.global_buffer.spatial_permutation}")
    lines.append(f"    temporal: {mapping.global_buffer.temporal_factors}")
    lines.append(f"    perm:     {mapping.global_buffer.temporal_permutation}")
    
    lines.append("\n  DRAM (all dataspaces):")
    lines.append(f"    temporal: {mapping.dram.temporal_factors}")
    lines.append(f"    perm:     {mapping.dram.temporal_permutation}")
    
    return "\n".join(lines)


def prettify_constraints(workload: Workload, architecture: Architecture) -> str:
    """Generate human-readable constraint description"""
    C, M, R, S, P, Q = workload.C, workload.M, workload.R, workload.S, workload.P, workload.Q
    
    lines = [
        "CONSTRAINTS",
        "=" * 60,
        "Factor Products (across all levels must equal workload dimensions):",
        f"  C = {C}  →  product of all C factors = {C}",
        f"              (PEAccuBuffer spatial C × rest C1)",
        "",
        f"  M = {M}  →  product of all M factors = {M}",
        f"              (PEInputBuffer M × GlobalBuffer M × DRAM M = {M})",
        "",
        f"  R = {R}  →  product of all R factors = {R}",
        f"              (PEWeightBuffer temporal R × rest R1)",
        "",
        f"  S = {S}  →  product of all S factors = {S}",
        f"              (PEWeightBuffer temporal S × rest S1)",
        "",
        f"  P = {P}  →  product of all P factors = {P}",
        f"              (PEWeightRegs P × GlobalBuffer P × DRAM P = {P})",
        "",
        f"  Q = {Q}  →  product of all Q factors = {Q}",
        f"              (PEWeightRegs Q × GlobalBuffer Q × DRAM Q = {Q})",
        "",
        "Spatial Fanout (per architecture mesh):",
        # The exact numeric mesh limit depends on the minimal architecture
        # computed by the evaluator from the template and mapping. We avoid
        # hard-coding a possibly incorrect bound here.
        "  PEAccuBuffer C × PEInputBuffer M × GlobalBuffer M must fit within the available PE mesh.",
        "",
        "Datatype Bindings (fixed):",
        "  PEWeightRegs:   Weights only",
        "  PEAccuBuffer:   Outputs only",
        "  PEWeightBuffer: Weights only",
        "  PEInputBuffer:  Inputs only",
        "  GlobalBuffer:   Inputs + Outputs",
        "  DRAM:           All dataspaces",
    ]
    
    return "\n".join(lines)


def build_full_context(bundle_path: str) -> str:
    """Build complete context string for LLM"""
    workload, architecture, mapping = parse_bundle(bundle_path)
    
    sections = [
        prettify_workload(workload),
        "",
        prettify_architecture(architecture),
        "",
        prettify_constraints(workload, architecture),
        "",
        prettify_mapping(mapping),
    ]
    
    return "\n".join(sections)


# ============================================================================
# Plotting
# ============================================================================

def plot_metrics(
    metrics_list: List[Metrics],
    output_path: str = "metrics.png",
    title: str = "Optimization Progress"
) -> None:
    """
    Plot metrics over generations.
    
    Args:
        metrics_list: List of Metrics objects (one per generation)
        output_path: Path to save plot
        title: Overall plot title
    """
    if not metrics_list:
        print("No metrics to plot")
        return
    
    energy = [m.energy for m in metrics_list]
    cycles = [m.cycles for m in metrics_list]
    area = [m.area for m in metrics_list]
    generations = list(range(len(metrics_list)))
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    axes[0].plot(generations, energy, 'b-o', linewidth=2, markersize=6)
    axes[0].set_xlabel('Generation', fontsize=11)
    axes[0].set_ylabel('Energy (pJ)', fontsize=11)
    axes[0].set_title('Energy', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(generations, cycles, 'r-s', linewidth=2, markersize=6)
    axes[1].set_xlabel('Generation', fontsize=11)
    axes[1].set_ylabel('Cycles', fontsize=11)
    axes[1].set_title('Execution Cycles', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(generations, area, 'g-^', linewidth=2, markersize=6)
    axes[2].set_xlabel('Generation', fontsize=11)
    axes[2].set_ylabel('Area (um²)', fontsize=11)
    axes[2].set_title('Hardware Area', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")


# ============================================================================
# Main (Demo/Test)
# ============================================================================

def main():
    """Demo: Parse bundle, evaluate mapping, print results"""
    
    bundle_path = "/workspaces/moham/output_eval6/medea/WL_C.3_M.64_R.7_S.7_N.1_P.112_Q.112_Wstride.2_Hstride.2_Wdilation.1_Hdilation.1/simba/pareto/medea.bundle.0.yaml"
    
    print("="*60)
    print("PARSING COMPLETE BUNDLE (STRICTLY TYPED)")
    print("="*60)
    
    workload, architecture, mapping = parse_bundle(bundle_path)
    
    print("\n" + prettify_workload(workload))
    print("\n" + prettify_architecture(architecture))
    print("\n" + prettify_constraints(workload, architecture))
    print("\n" + prettify_mapping(mapping))
    
    print("\n" + "="*60)
    print("TESTING BUNDLE RECONSTRUCTION")
    print("="*60)
    
    # Test bundle reconstruction
    reconstructed_bundle = build_bundle(workload, architecture, mapping)
    
    # Save reconstructed bundle for comparison
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.safe_dump(reconstructed_bundle, f, sort_keys=False)
        temp_path = f.name
    
    print(f"Reconstructed bundle saved to: {temp_path}")
    
    # Compare key sections
    print("\nArchitecture comparison:")
    print(f"  Original template: simba")
    print(f"  Reconstructed template: {reconstructed_bundle['meta']['template']}")
    print(f"  Original arch name: System")
    print(f"  Reconstructed arch name: {reconstructed_bundle['architecture']['name']}")
    print(f"  Original datawidth: 8")
    print(f"  Reconstructed datawidth: {reconstructed_bundle['architecture']['attributes']['datawidth']}")
    
    print("\n" + "="*60)
    print("EVALUATING REFERENCE MAPPING")
    print("="*60)
    
    result = evaluate_mapping(workload, architecture, mapping)
    
    if isinstance(result, Metrics):
        print(f"\n✓ Evaluation successful")
        print(f"  Energy:  {result.energy:,.2f} pJ")
        print(f"  Cycles:  {result.cycles:,}")
        print(f"  Area:    {result.area:,.2f} um²")
        print(f"\n  Expected: E=263,100,411.53 C=4,917,248 A=57,011.24")
    else:
        print(f"\n✗ Evaluation failed")
        print(result.to_string())
    
    # Test plotting
    print("\n" + "="*60)
    print("TESTING PLOT FUNCTION")
    print("="*60)
    
    mock_metrics = [
        Metrics(energy=500000000, cycles=3000000, area=40000),
        Metrics(energy=480000000, cycles=3900000, area=49500),
        Metrics(energy=470000000, cycles=3850000, area=49550),
        Metrics(energy=460000000, cycles=3820000, area=48550),
        Metrics(energy=455000000, cycles=3780000, area=48500),
    ]
    plot_metrics(mock_metrics, "test_metrics.png", "Mock Optimization")
    
    # Clean up temp file
    import os
    os.unlink(temp_path)


if __name__ == "__main__":
    main()
