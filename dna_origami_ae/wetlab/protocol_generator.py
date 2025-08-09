"""Protocol generator for wet-lab DNA origami assembly."""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

from ..models.origami_structure import OrigamiStructure
from ..models.dna_sequence import DNASequence


class LabEquipment(Enum):
    """Standard laboratory equipment configurations."""
    STANDARD_BIO_LAB = "standard_bio_lab"
    AUTOMATED_SYSTEM = "automated_system"
    MINIMAL_SETUP = "minimal_setup"
    HIGH_THROUGHPUT = "high_throughput"


@dataclass
class ProtocolStep:
    """Single step in a laboratory protocol."""
    step_number: int
    title: str
    description: str
    duration_minutes: float
    temperature: Optional[float] = None
    equipment_needed: List[str] = field(default_factory=list)
    reagents: List[Dict[str, Any]] = field(default_factory=list)
    safety_warnings: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    troubleshooting: Dict[str, str] = field(default_factory=dict)


@dataclass
class AssemblyConditions:
    """Conditions for DNA origami assembly."""
    annealing_temperature_start: float = 95.0  # °C
    annealing_temperature_end: float = 20.0    # °C
    cooling_rate: float = 1.0                  # °C/hour
    mg_concentration: float = 12.5             # mM
    nacl_concentration: float = 50.0           # mM
    tris_concentration: float = 10.0           # mM
    ph: float = 8.0
    total_volume: float = 100.0                # μL
    staple_scaffold_ratio: int = 10            # 10:1 excess


class ProtocolGenerator:
    """Generate comprehensive laboratory protocols for DNA origami assembly."""
    
    def __init__(self, equipment: LabEquipment = LabEquipment.STANDARD_BIO_LAB,
                 safety_level: str = "BSL1"):
        """Initialize protocol generator."""
        self.equipment = equipment
        self.safety_level = safety_level
        self.protocol_counter = 0
        
    def generate_protocol(self, structure: OrigamiStructure, 
                         scale: str = "single_tube",
                         replicates: int = 1,
                         assembly_conditions: Optional[AssemblyConditions] = None) -> 'WetLabProtocol':
        """Generate complete protocol for origami assembly."""
        
        conditions = assembly_conditions or AssemblyConditions()
        self.protocol_counter += 1
        
        protocol = WetLabProtocol(
            protocol_id=f"DNA_ORIGAMI_{self.protocol_counter:03d}",
            structure_name=structure.name,
            scale=scale,
            replicates=replicates,
            conditions=conditions,
            equipment=self.equipment,
            safety_level=self.safety_level
        )
        
        # Generate protocol steps
        steps = []
        
        # Step 1: Preparation
        steps.extend(self._generate_preparation_steps(structure, conditions))
        
        # Step 2: Solution preparation
        steps.extend(self._generate_solution_prep_steps(structure, conditions, replicates))
        
        # Step 3: Assembly reaction
        steps.extend(self._generate_assembly_steps(structure, conditions))
        
        # Step 4: Purification
        steps.extend(self._generate_purification_steps(structure))
        
        # Step 5: Quality control
        steps.extend(self._generate_qc_steps(structure))
        
        protocol.steps = steps
        protocol.estimated_time_hours = sum(step.duration_minutes for step in steps) / 60
        
        return protocol
    
    def _generate_preparation_steps(self, structure: OrigamiStructure, 
                                   conditions: AssemblyConditions) -> List[ProtocolStep]:
        """Generate preparation steps."""
        steps = []
        
        # Equipment preparation
        steps.append(ProtocolStep(
            step_number=1,
            title="Equipment Preparation",
            description=f"""
            1. Set up thermal cycler with the following program:
               - Initial denaturation: {conditions.annealing_temperature_start}°C for 5 min
               - Cooling ramp: {conditions.annealing_temperature_start}°C to {conditions.annealing_temperature_end}°C 
                 at {conditions.cooling_rate}°C/hour
               - Hold at {conditions.annealing_temperature_end}°C
            
            2. Prepare microcentrifuge tubes (200 μL capacity)
            3. Prepare ice bucket for reagent handling
            4. Ensure pipettes are calibrated (P10, P200, P1000)
            """,
            duration_minutes=15,
            equipment_needed=["thermal_cycler", "microcentrifuge_tubes", "pipettes", "ice_bucket"],
            safety_warnings=["Ensure thermal cycler is properly calibrated", "Use proper PPE"]
        ))
        
        # Buffer preparation
        steps.append(ProtocolStep(
            step_number=2,
            title="Buffer Preparation", 
            description=f"""
            Prepare 10x Origami Buffer:
            - Tris-HCl pH {conditions.ph}: {conditions.tris_concentration * 10} mM
            - MgCl2: {conditions.mg_concentration * 10} mM  
            - NaCl: {conditions.nacl_concentration * 10} mM
            
            Prepare working solution (1x) by diluting 10x stock 1:10 with nuclease-free water.
            Store on ice until use.
            """,
            duration_minutes=20,
            temperature=4.0,
            equipment_needed=["balance", "pH_meter", "magnetic_stirrer"],
            reagents=[
                {"name": "Tris-HCl", "amount": f"{conditions.tris_concentration * 10} mM", "supplier": "Sigma"},
                {"name": "MgCl2", "amount": f"{conditions.mg_concentration * 10} mM", "supplier": "Sigma"},
                {"name": "NaCl", "amount": f"{conditions.nacl_concentration * 10} mM", "supplier": "Sigma"},
                {"name": "Nuclease-free water", "amount": "As needed", "supplier": "Any"}
            ],
            safety_warnings=["Handle chemicals with appropriate PPE"]
        ))
        
        return steps
    
    def _generate_solution_prep_steps(self, structure: OrigamiStructure,
                                    conditions: AssemblyConditions,
                                    replicates: int) -> List[ProtocolStep]:
        """Generate solution preparation steps."""
        steps = []
        
        # Calculate concentrations
        scaffold_conc = 20.0  # nM
        staple_conc = scaffold_conc * conditions.staple_scaffold_ratio
        
        steps.append(ProtocolStep(
            step_number=len(steps) + 3,
            title="DNA Solution Preparation",
            description=f"""
            Prepare the following solutions for {replicates} reaction(s):
            
            1. Scaffold solution:
               - {structure.scaffold.sequence.name}: {scaffold_conc} nM final concentration
               - Volume needed: {conditions.total_volume * 0.1 * replicates:.1f} μL
            
            2. Staple mix:
               - Each staple: {staple_conc / len(structure.staples):.1f} nM final concentration
               - Total staples: {len(structure.staples)}
               - Combine all staples in equal molar ratios
               - Volume needed: {conditions.total_volume * 0.8 * replicates:.1f} μL
            
            3. Buffer solution:
               - 1x Origami Buffer
               - Volume needed: {conditions.total_volume * 0.1 * replicates:.1f} μL
            
            Keep all solutions on ice.
            """,
            duration_minutes=30,
            temperature=4.0,
            equipment_needed=["microcentrifuge_tubes", "pipettes", "ice_bucket"],
            reagents=[
                {"name": "Scaffold DNA", "concentration": f"{scaffold_conc} nM", "amount": f"{conditions.total_volume * 0.1:.1f} μL"},
                {"name": "Staple mix", "concentration": f"{staple_conc} nM total", "amount": f"{conditions.total_volume * 0.8:.1f} μL"},
                {"name": "1x Origami Buffer", "amount": f"{conditions.total_volume * 0.1:.1f} μL"}
            ],
            success_criteria=[
                "All solutions are clear and well-mixed",
                "Solutions maintained at 4°C",
                "Concentrations calculated correctly"
            ]
        ))
        
        return steps
    
    def _generate_assembly_steps(self, structure: OrigamiStructure,
                               conditions: AssemblyConditions) -> List[ProtocolStep]:
        """Generate assembly reaction steps."""
        steps = []
        
        annealing_time = (conditions.annealing_temperature_start - conditions.annealing_temperature_end) / conditions.cooling_rate
        
        steps.append(ProtocolStep(
            step_number=len(steps) + 4,
            title="Assembly Reaction Setup",
            description=f"""
            In a 200 μL PCR tube, combine:
            1. Scaffold DNA solution: {conditions.total_volume * 0.1:.1f} μL
            2. Staple mix: {conditions.total_volume * 0.8:.1f} μL  
            3. 1x Origami Buffer: {conditions.total_volume * 0.1:.1f} μL
            
            Total volume: {conditions.total_volume} μL
            
            Mix gently by pipetting up and down 3 times.
            Avoid creating bubbles.
            """,
            duration_minutes=5,
            temperature=4.0,
            equipment_needed=["pcr_tubes", "pipettes"],
            success_criteria=[
                "Solution is homogeneous", 
                "No visible bubbles",
                f"Total volume is {conditions.total_volume} μL"
            ],
            troubleshooting={
                "Solution appears cloudy": "Check pH of buffer, may need adjustment",
                "Visible precipitate": "Reduce MgCl2 concentration by 25%"
            }
        ))
        
        steps.append(ProtocolStep(
            step_number=len(steps) + 5,
            title="Thermal Annealing",
            description=f"""
            1. Place tube in thermal cycler
            2. Run annealing program:
               - Initial denaturation: {conditions.annealing_temperature_start}°C for 5 min
               - Slow cooling: {conditions.annealing_temperature_start}°C → {conditions.annealing_temperature_end}°C
                 at {conditions.cooling_rate}°C/hour
               - Hold at {conditions.annealing_temperature_end}°C for 30 min
            
            Total annealing time: ~{annealing_time + 0.6:.1f} hours
            
            DO NOT open thermal cycler during the run.
            """,
            duration_minutes=(annealing_time + 0.6) * 60,
            equipment_needed=["thermal_cycler"],
            safety_warnings=["Do not interrupt thermal cycling program"],
            success_criteria=[
                "Program completes without errors",
                "Sample remains liquid (no evaporation)"
            ],
            troubleshooting={
                "Evaporation occurred": "Check seal on tube, use mineral oil overlay if needed",
                "Program error": "Restart with fresh sample, check thermal cycler calibration"
            }
        ))
        
        return steps
    
    def _generate_purification_steps(self, structure: OrigamiStructure) -> List[ProtocolStep]:
        """Generate purification steps."""
        steps = []
        
        steps.append(ProtocolStep(
            step_number=len(steps) + 6,
            title="Gel Filtration Purification",
            description="""
            1. Prepare 0.5% agarose gel in 1x TAE buffer + 10 mM MgCl2
            2. Load assembled origami sample (do not add loading dye)
            3. Run at 100V for 2 hours at 4°C
            4. Stain with SYBR Safe (1:10,000) for 30 minutes
            5. Visualize on blue light transilluminator
            6. Excise band corresponding to folded origami
            7. Purify DNA from gel using standard gel extraction kit
            
            Expected band: Slower migration than scaffold alone
            """,
            duration_minutes=180,
            temperature=4.0,
            equipment_needed=["gel_electrophoresis", "transilluminator", "gel_extraction_kit"],
            reagents=[
                {"name": "Agarose", "amount": "0.5% w/v", "supplier": "Any"},
                {"name": "TAE buffer", "amount": "1x", "supplier": "Any"},
                {"name": "MgCl2", "amount": "10 mM", "supplier": "Sigma"},
                {"name": "SYBR Safe", "amount": "1:10,000", "supplier": "Invitrogen"}
            ],
            success_criteria=[
                "Clear band separation visible",
                "Folded origami band is distinct from excess staples",
                "Gel extraction yields >50% recovery"
            ],
            troubleshooting={
                "No band visible": "Check SYBR Safe concentration and UV settings",
                "Multiple bands": "Partial folding - optimize annealing conditions",
                "Low yield after extraction": "Use fresh gel extraction kit, minimize gel volume"
            }
        ))
        
        return steps
    
    def _generate_qc_steps(self, structure: OrigamiStructure) -> List[ProtocolStep]:
        """Generate quality control steps."""
        steps = []
        
        steps.append(ProtocolStep(
            step_number=len(steps) + 7,
            title="Quality Control Analysis",
            description="""
            Perform the following quality control analyses:
            
            1. UV-Vis Spectroscopy:
               - Measure A260 to determine concentration
               - Check A260/A280 ratio (should be ~1.8-2.0)
               - Calculate yield based on theoretical scaffold content
            
            2. Dynamic Light Scattering (optional):
               - Measure size distribution
               - Expected size: Based on design dimensions
            
            3. Atomic Force Microscopy (recommended):
               - Deposit 5 μL sample on freshly cleaved mica
               - Rinse with buffer, then water
               - Air dry and image in tapping mode
               - Verify origami shape and structure
            
            4. Storage:
               - Store at 4°C in origami buffer
               - Add 50% glycerol for long-term storage at -20°C
            """,
            duration_minutes=120,
            equipment_needed=["uv_vis_spectrometer", "DLS", "AFM", "mica_substrates"],
            success_criteria=[
                "A260/A280 ratio between 1.8-2.0",
                "AFM images show expected origami structure",
                "Yield >10% based on scaffold input"
            ],
            troubleshooting={
                "Low A260/A280": "Possible protein contamination - repurify",
                "Aggregation in DLS": "Dilute sample, check buffer conditions",
                "Poor AFM structure": "Optimize surface preparation and imaging conditions"
            }
        ))
        
        return steps
        

class WetLabProtocol:
    """Complete wet-lab protocol for DNA origami assembly."""
    
    def __init__(self, protocol_id: str, structure_name: str, scale: str,
                 replicates: int, conditions: AssemblyConditions,
                 equipment: LabEquipment, safety_level: str):
        """Initialize protocol."""
        self.protocol_id = protocol_id
        self.structure_name = structure_name
        self.scale = scale
        self.replicates = replicates
        self.conditions = conditions
        self.equipment = equipment
        self.safety_level = safety_level
        self.steps: List[ProtocolStep] = []
        self.estimated_time_hours = 0.0
        self.created_timestamp = time.time()
        
    def to_pdf(self, filename: str) -> None:
        """Export protocol to PDF format."""
        # In a real implementation, would use a library like reportlab
        print(f"PDF export to {filename} - Implementation needed")
        
    def to_opentrons(self, filename: str) -> None:
        """Export protocol for Opentrons robot."""
        opentrons_protocol = {
            "metadata": {
                "protocolName": f"DNA Origami Assembly - {self.structure_name}",
                "author": "DNA-Origami-AutoEncoder",
                "description": f"Automated assembly protocol for {self.structure_name}",
                "apiLevel": "2.13"
            },
            "protocol_steps": []
        }
        
        for step in self.steps:
            if "pipette" in step.description.lower() or "mix" in step.description.lower():
                opentrons_protocol["protocol_steps"].append({
                    "step": step.step_number,
                    "title": step.title,
                    "description": step.description,
                    "reagents": step.reagents,
                    "equipment": step.equipment_needed
                })
        
        with open(filename, 'w') as f:
            json.dump(opentrons_protocol, f, indent=2)
    
    def to_benchling(self) -> Dict[str, Any]:
        """Export protocol for Benchling integration."""
        benchling_format = {
            "name": f"DNA Origami Assembly - {self.structure_name}",
            "description": f"Assembly protocol for {self.structure_name} DNA origami",
            "protocolId": self.protocol_id,
            "steps": []
        }
        
        for step in self.steps:
            benchling_step = {
                "stepNumber": step.step_number,
                "title": step.title,
                "description": step.description,
                "duration": step.duration_minutes,
                "temperature": step.temperature,
                "reagents": step.reagents,
                "equipment": step.equipment_needed,
                "safetyWarnings": step.safety_warnings
            }
            benchling_format["steps"].append(benchling_step)
        
        return benchling_format
    
    def generate_materials_list(self, include_prices: bool = False,
                              vendors: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive materials and reagents list."""
        if vendors is None:
            vendors = ["IDT", "NEB", "Sigma", "Invitrogen"]
        
        materials = {
            "reagents": {},
            "equipment": set(),
            "consumables": set()
        }
        
        # Collect all reagents
        for step in self.steps:
            for reagent in step.reagents:
                name = reagent["name"]
                if name not in materials["reagents"]:
                    materials["reagents"][name] = {
                        "amount_needed": reagent.get("amount", "TBD"),
                        "supplier": reagent.get("supplier", "Any"),
                        "catalog_number": reagent.get("catalog", "TBD"),
                        "estimated_price": reagent.get("price", "Contact supplier") if include_prices else None
                    }
            
            # Collect equipment
            materials["equipment"].update(step.equipment_needed)
        
        # Add standard consumables
        materials["consumables"].update([
            "microcentrifuge_tubes_200ul",
            "pcr_tubes",
            "pipette_tips_p10",
            "pipette_tips_p200", 
            "pipette_tips_p1000",
            "gloves_nitrile",
            "mica_substrates"
        ])
        
        # Convert sets to lists for JSON serialization
        materials["equipment"] = list(materials["equipment"])
        materials["consumables"] = list(materials["consumables"])
        
        return materials
    
    def get_protocol_summary(self) -> Dict[str, Any]:
        """Get protocol summary information."""
        return {
            "protocol_id": self.protocol_id,
            "structure_name": self.structure_name,
            "total_steps": len(self.steps),
            "estimated_time_hours": self.estimated_time_hours,
            "scale": self.scale,
            "replicates": self.replicates,
            "equipment_level": self.equipment.value,
            "safety_level": self.safety_level,
            "assembly_conditions": {
                "annealing_temp_range": f"{self.conditions.annealing_temperature_start}-{self.conditions.annealing_temperature_end}°C",
                "cooling_rate": f"{self.conditions.cooling_rate}°C/hour",
                "buffer_conditions": f"{self.conditions.mg_concentration}mM Mg2+, {self.conditions.nacl_concentration}mM NaCl"
            },
            "created": time.ctime(self.created_timestamp)
        }