"""Plate layout designer for high-throughput DNA origami experiments."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from ..models.origami_structure import OrigamiStructure
from ..models.dna_sequence import DNASequence


class PlateFormat(Enum):
    """Standard plate formats."""
    PLATE_96 = (8, 12)  # 8 rows, 12 columns
    PLATE_384 = (16, 24)  # 16 rows, 24 columns
    PLATE_1536 = (32, 48)  # 32 rows, 48 columns


@dataclass
class WellContent:
    """Contents of a single well."""
    well_id: str
    experiment_type: str
    scaffold_conc: float = 0.0  # nM
    staple_conc: float = 0.0    # nM
    buffer_volume: float = 0.0  # μL
    total_volume: float = 100.0 # μL
    temperature: Optional[float] = None
    notes: str = ""
    replicate_group: Optional[str] = None
    expected_outcome: str = ""


@dataclass 
class WellPlate96:
    """96-well plate layout and contents."""
    plate_id: str
    rows: int = 8
    columns: int = 12
    well_contents: Dict[str, WellContent] = field(default_factory=dict)
    plate_type: str = "standard_96_well"
    created_timestamp: float = field(default_factory=lambda: __import__('time').time())
    
    def get_well_id(self, row: int, col: int) -> str:
        """Convert row/column to well ID (e.g., A01, B12)."""
        row_letter = chr(ord('A') + row)
        return f"{row_letter}{col+1:02d}"
    
    def get_row_col(self, well_id: str) -> Tuple[int, int]:
        """Convert well ID to row/column indices."""
        row = ord(well_id[0]) - ord('A')
        col = int(well_id[1:]) - 1
        return row, col
    
    def add_well_content(self, well_id: str, content: WellContent) -> None:
        """Add content to specific well."""
        self.well_contents[well_id] = content
    
    def get_well_content(self, well_id: str) -> Optional[WellContent]:
        """Get content of specific well."""
        return self.well_contents.get(well_id)
    
    def export_to_csv(self, filename: str) -> None:
        """Export plate layout to CSV format."""
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Well_ID', 'Experiment_Type', 'Scaffold_Conc_nM', 
                           'Staple_Conc_nM', 'Buffer_Volume_uL', 'Total_Volume_uL',
                           'Temperature_C', 'Replicate_Group', 'Notes'])
            
            # Data rows
            for well_id in sorted(self.well_contents.keys()):
                content = self.well_contents[well_id]
                writer.writerow([
                    well_id,
                    content.experiment_type,
                    content.scaffold_conc,
                    content.staple_conc,
                    content.buffer_volume,
                    content.total_volume,
                    content.temperature or '',
                    content.replicate_group or '',
                    content.notes
                ])
    
    def visualize_layout(self) -> str:
        """Create ASCII visualization of plate layout."""
        visualization = f"Plate: {self.plate_id}\n"
        visualization += "   " + "".join(f"{i+1:3d}" for i in range(self.columns)) + "\n"
        
        for row in range(self.rows):
            row_letter = chr(ord('A') + row)
            line = f"{row_letter}  "
            
            for col in range(self.columns):
                well_id = self.get_well_id(row, col)
                if well_id in self.well_contents:
                    # Use first letter of experiment type
                    exp_type = self.well_contents[well_id].experiment_type
                    symbol = exp_type[0].upper() if exp_type else 'X'
                else:
                    symbol = '.'  # Empty well
                line += f" {symbol} "
            
            visualization += line + "\n"
        
        return visualization


class PlateMaps:
    """Common plate mapping strategies."""
    
    @staticmethod
    def create_concentration_series(scaffold_concs: List[float],
                                  staple_concs: List[float],
                                  replicates: int = 3) -> List[Dict[str, Any]]:
        """Create concentration series experimental design."""
        experiments = []
        
        for scaffold_conc in scaffold_concs:
            for staple_conc in staple_concs:
                for rep in range(replicates):
                    experiments.append({
                        'scaffold_conc': scaffold_conc,
                        'staple_conc': staple_conc,
                        'replicate': rep + 1,
                        'experiment_type': 'concentration_series'
                    })
        
        return experiments
    
    @staticmethod
    def create_temperature_series(temperatures: List[float],
                                scaffold_conc: float = 20.0,
                                staple_conc: float = 200.0,
                                replicates: int = 3) -> List[Dict[str, Any]]:
        """Create temperature series experimental design."""
        experiments = []
        
        for temp in temperatures:
            for rep in range(replicates):
                experiments.append({
                    'temperature': temp,
                    'scaffold_conc': scaffold_conc,
                    'staple_conc': staple_conc,
                    'replicate': rep + 1,
                    'experiment_type': 'temperature_series'
                })
        
        return experiments
    
    @staticmethod
    def create_buffer_optimization(mg_concs: List[float],
                                 nacl_concs: List[float],
                                 replicates: int = 2) -> List[Dict[str, Any]]:
        """Create buffer optimization experimental design."""
        experiments = []
        
        for mg_conc in mg_concs:
            for nacl_conc in nacl_concs:
                for rep in range(replicates):
                    experiments.append({
                        'mg_concentration': mg_conc,
                        'nacl_concentration': nacl_conc,
                        'replicate': rep + 1,
                        'experiment_type': 'buffer_optimization'
                    })
        
        return experiments


class PlateDesigner:
    """Design experimental plate layouts for DNA origami studies."""
    
    def __init__(self, plate_format: PlateFormat = PlateFormat.PLATE_96):
        """Initialize plate designer."""
        self.plate_format = plate_format
        self.rows, self.columns = plate_format.value
        self.total_wells = self.rows * self.columns
        self.plate_counter = 0
    
    def design_concentration_study_plate(self, 
                                       structures: List[OrigamiStructure],
                                       scaffold_concs: List[float] = None,
                                       staple_ratios: List[int] = None,
                                       replicates: int = 3) -> WellPlate96:
        """Design plate for concentration studies."""
        if scaffold_concs is None:
            scaffold_concs = [10.0, 20.0, 50.0]  # nM
        if staple_ratios is None:
            staple_ratios = [5, 10, 20]  # fold excess
        
        self.plate_counter += 1
        plate = WellPlate96(
            plate_id=f"CONC_STUDY_{self.plate_counter:03d}",
            rows=self.rows,
            columns=self.columns
        )
        
        well_index = 0
        
        for structure in structures:
            for scaffold_conc in scaffold_concs:
                for ratio in staple_ratios:
                    staple_conc = scaffold_conc * ratio
                    
                    for rep in range(replicates):
                        if well_index >= self.total_wells:
                            print(f"Warning: Experiment exceeds plate capacity ({self.total_wells} wells)")
                            break
                        
                        row = well_index // self.columns
                        col = well_index % self.columns
                        well_id = plate.get_well_id(row, col)
                        
                        content = WellContent(
                            well_id=well_id,
                            experiment_type="concentration_study",
                            scaffold_conc=scaffold_conc,
                            staple_conc=staple_conc,
                            buffer_volume=80.0,
                            total_volume=100.0,
                            replicate_group=f"{structure.name}_S{scaffold_conc}_R{ratio}_rep{rep+1}",
                            notes=f"Structure: {structure.name}, Ratio: {ratio}:1",
                            expected_outcome="folded_origami"
                        )
                        
                        plate.add_well_content(well_id, content)
                        well_index += 1
        
        return plate
    
    def design_temperature_optimization_plate(self,
                                            structure: OrigamiStructure,
                                            temperatures: List[float] = None,
                                            annealing_rates: List[float] = None,
                                            replicates: int = 4) -> WellPlate96:
        """Design plate for temperature optimization."""
        if temperatures is None:
            temperatures = [45.0, 50.0, 55.0, 60.0, 65.0]  # °C
        if annealing_rates is None:
            annealing_rates = [0.5, 1.0, 2.0]  # °C/hour
        
        self.plate_counter += 1
        plate = WellPlate96(
            plate_id=f"TEMP_OPT_{self.plate_counter:03d}",
            rows=self.rows,
            columns=self.columns
        )
        
        # Add controls
        self._add_controls(plate, structure)
        
        well_index = 12  # Start after controls (first 12 wells)
        
        for temp in temperatures:
            for rate in annealing_rates:
                for rep in range(replicates):
                    if well_index >= self.total_wells:
                        print(f"Warning: Experiment exceeds plate capacity")
                        break
                    
                    row = well_index // self.columns
                    col = well_index % self.columns
                    well_id = plate.get_well_id(row, col)
                    
                    content = WellContent(
                        well_id=well_id,
                        experiment_type="temperature_optimization",
                        scaffold_conc=20.0,
                        staple_conc=200.0,
                        buffer_volume=80.0,
                        total_volume=100.0,
                        temperature=temp,
                        replicate_group=f"T{temp}_R{rate}_rep{rep+1}",
                        notes=f"Annealing temp: {temp}°C, Rate: {rate}°C/h",
                        expected_outcome="optimal_folding"
                    )
                    
                    plate.add_well_content(well_id, content)
                    well_index += 1
        
        return plate
    
    def design_buffer_optimization_plate(self,
                                       structure: OrigamiStructure,
                                       mg_concentrations: List[float] = None,
                                       salt_concentrations: List[float] = None,
                                       ph_values: List[float] = None,
                                       replicates: int = 3) -> WellPlate96:
        """Design plate for buffer optimization."""
        if mg_concentrations is None:
            mg_concentrations = [5.0, 10.0, 15.0, 20.0]  # mM
        if salt_concentrations is None:
            salt_concentrations = [25.0, 50.0, 100.0]  # mM NaCl
        if ph_values is None:
            ph_values = [7.5, 8.0, 8.5]
        
        self.plate_counter += 1
        plate = WellPlate96(
            plate_id=f"BUFFER_OPT_{self.plate_counter:03d}",
            rows=self.rows,
            columns=self.columns
        )
        
        well_index = 0
        
        for mg_conc in mg_concentrations:
            for salt_conc in salt_concentrations:
                for ph in ph_values:
                    for rep in range(replicates):
                        if well_index >= self.total_wells:
                            print(f"Warning: Experiment exceeds plate capacity")
                            break
                        
                        row = well_index // self.columns
                        col = well_index % self.columns
                        well_id = plate.get_well_id(row, col)
                        
                        content = WellContent(
                            well_id=well_id,
                            experiment_type="buffer_optimization",
                            scaffold_conc=20.0,
                            staple_conc=200.0,
                            buffer_volume=80.0,
                            total_volume=100.0,
                            replicate_group=f"Mg{mg_conc}_NaCl{salt_conc}_pH{ph}_rep{rep+1}",
                            notes=f"Mg2+: {mg_conc}mM, NaCl: {salt_conc}mM, pH: {ph}",
                            expected_outcome="optimized_assembly"
                        )
                        
                        plate.add_well_content(well_id, content)
                        well_index += 1
        
        return plate
    
    def design_multi_structure_comparison_plate(self,
                                              structures: List[OrigamiStructure],
                                              conditions: Dict[str, Any] = None,
                                              replicates: int = 6) -> WellPlate96:
        """Design plate for comparing multiple structures."""
        if conditions is None:
            conditions = {
                'scaffold_conc': 20.0,
                'staple_ratio': 10,
                'temperature': 55.0
            }
        
        self.plate_counter += 1
        plate = WellPlate96(
            plate_id=f"MULTI_STRUCT_{self.plate_counter:03d}",
            rows=self.rows,
            columns=self.columns
        )
        
        well_index = 0
        
        for structure in structures:
            for rep in range(replicates):
                if well_index >= self.total_wells:
                    print(f"Warning: Experiment exceeds plate capacity")
                    break
                
                row = well_index // self.columns
                col = well_index % self.columns
                well_id = plate.get_well_id(row, col)
                
                staple_conc = conditions['scaffold_conc'] * conditions['staple_ratio']
                
                content = WellContent(
                    well_id=well_id,
                    experiment_type="structure_comparison",
                    scaffold_conc=conditions['scaffold_conc'],
                    staple_conc=staple_conc,
                    buffer_volume=80.0,
                    total_volume=100.0,
                    temperature=conditions.get('temperature'),
                    replicate_group=f"{structure.name}_rep{rep+1}",
                    notes=f"Structure: {structure.name}",
                    expected_outcome="structure_specific_assembly"
                )
                
                plate.add_well_content(well_id, content)
                well_index += 1
        
        # Fill remaining wells with controls
        self._add_remaining_controls(plate, well_index, conditions)
        
        return plate
    
    def _add_controls(self, plate: WellPlate96, structure: OrigamiStructure) -> None:
        """Add standard control wells."""
        controls = [
            ("negative_no_scaffold", 0.0, 200.0, "No scaffold control"),
            ("negative_no_staples", 20.0, 0.0, "No staples control"), 
            ("positive_standard", 20.0, 200.0, "Standard conditions"),
            ("blank_buffer_only", 0.0, 0.0, "Buffer only blank")
        ]
        
        for i, (ctrl_type, scaffold_conc, staple_conc, notes) in enumerate(controls):
            if i < 12:  # First row
                well_id = plate.get_well_id(0, i)
                content = WellContent(
                    well_id=well_id,
                    experiment_type=ctrl_type,
                    scaffold_conc=scaffold_conc,
                    staple_conc=staple_conc,
                    buffer_volume=80.0,
                    total_volume=100.0,
                    replicate_group="controls",
                    notes=notes,
                    expected_outcome="control_result"
                )
                plate.add_well_content(well_id, content)
    
    def _add_remaining_controls(self, plate: WellPlate96, start_index: int,
                              conditions: Dict[str, Any]) -> None:
        """Fill remaining wells with additional controls."""
        remaining_wells = self.total_wells - start_index
        
        if remaining_wells > 0:
            # Add replicate controls
            for i in range(min(remaining_wells, 8)):
                row = (start_index + i) // self.columns
                col = (start_index + i) % self.columns
                well_id = plate.get_well_id(row, col)
                
                content = WellContent(
                    well_id=well_id,
                    experiment_type="replicate_control",
                    scaffold_conc=conditions['scaffold_conc'],
                    staple_conc=conditions['scaffold_conc'] * conditions['staple_ratio'],
                    buffer_volume=80.0,
                    total_volume=100.0,
                    temperature=conditions.get('temperature'),
                    replicate_group="additional_controls",
                    notes="Additional control replicate",
                    expected_outcome="standard_assembly"
                )
                plate.add_well_content(well_id, content)
    
    def generate_plate_reader_protocol(self, plate: WellPlate96,
                                     measurement_type: str = "fluorescence") -> Dict[str, Any]:
        """Generate plate reader protocol for automated measurements."""
        protocol = {
            "protocol_name": f"Automated_Analysis_{plate.plate_id}",
            "plate_id": plate.plate_id,
            "measurement_type": measurement_type,
            "settings": {},
            "kinetic_measurements": True,
            "measurement_schedule": []
        }
        
        if measurement_type == "fluorescence":
            protocol["settings"] = {
                "excitation_wavelength": 485,
                "emission_wavelength": 520,
                "gain": 100,
                "integration_time": 100,  # ms
                "number_of_flashes": 5
            }
        elif measurement_type == "absorbance":
            protocol["settings"] = {
                "wavelengths": [260, 280, 320],
                "pathlength_correction": True
            }
        
        # Add measurement timepoints
        timepoints = [0, 30, 60, 120, 240, 480, 1440]  # minutes
        for timepoint in timepoints:
            protocol["measurement_schedule"].append({
                "time_minutes": timepoint,
                "wells_to_measure": list(plate.well_contents.keys()),
                "expected_signal": "structure_dependent"
            })
        
        return protocol
    
    def estimate_reagent_requirements(self, plates: List[WellPlate96]) -> Dict[str, float]:
        """Estimate total reagent requirements for multiple plates."""
        requirements = {
            "scaffold_dna_nmol": 0.0,
            "total_staples_nmol": 0.0,
            "buffer_ml": 0.0,
            "plates_needed": len(plates)
        }
        
        for plate in plates:
            for well_content in plate.well_contents.values():
                # Calculate nmoles needed (concentration * volume)
                scaffold_nmol = (well_content.scaffold_conc * well_content.total_volume) / 1000
                staples_nmol = (well_content.staple_conc * well_content.total_volume) / 1000
                buffer_ml = well_content.buffer_volume / 1000
                
                requirements["scaffold_dna_nmol"] += scaffold_nmol
                requirements["total_staples_nmol"] += staples_nmol  
                requirements["buffer_ml"] += buffer_ml
        
        # Add 20% excess for pipetting losses
        for key in ["scaffold_dna_nmol", "total_staples_nmol", "buffer_ml"]:
            requirements[key] *= 1.2
        
        return requirements
    
    def export_plate_maps(self, plates: List[WellPlate96], output_dir: str) -> List[str]:
        """Export all plate maps to files."""
        exported_files = []
        
        for plate in plates:
            # CSV export
            csv_file = f"{output_dir}/{plate.plate_id}_layout.csv"
            plate.export_to_csv(csv_file)
            exported_files.append(csv_file)
            
            # JSON export  
            json_file = f"{output_dir}/{plate.plate_id}_layout.json"
            plate_data = {
                "plate_id": plate.plate_id,
                "format": f"{plate.rows}x{plate.columns}",
                "well_contents": {
                    well_id: {
                        "experiment_type": content.experiment_type,
                        "scaffold_conc": content.scaffold_conc,
                        "staple_conc": content.staple_conc,
                        "total_volume": content.total_volume,
                        "temperature": content.temperature,
                        "replicate_group": content.replicate_group,
                        "notes": content.notes
                    }
                    for well_id, content in plate.well_contents.items()
                }
            }
            
            with open(json_file, 'w') as f:
                json.dump(plate_data, f, indent=2)
            exported_files.append(json_file)
        
        return exported_files