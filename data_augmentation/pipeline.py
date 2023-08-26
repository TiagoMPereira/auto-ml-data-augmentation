from data_augmentation.conditions import create_conditions
from data_augmentation.diagnosis import diagnostic
from data_augmentation.metadata import SDVMetadata
from data_augmentation.synthesizers import (CTGAN, FASTML, TVAE, CopulaGAN,
                                      GaussianCopula)
import pandas as pd


def synthesize_data(
    data: pd.DataFrame, target_name: str, synthesizer_name: str
):
    metadata = SDVMetadata()
    metadata.create_from_df(data)
    diagnosis = diagnostic(data, target_name)
    classes_to_generate = diagnosis["rows_to_generate"]
    
    synthesizers = {
        "ctgan": CTGAN(metadata.metadata),
        "fastml": FASTML(metadata.metadata),
        "tvae": TVAE(metadata.metadata),
        "copulagan": CopulaGAN(metadata.metadata),
        "gaussiancopula": GaussianCopula(metadata.metadata),
    }

    synt = synthesizers.get(synthesizer_name, None)

    if not synt:
        raise ValueError(f"Synthesizer {synthesizer_name} is not available")

    synt.fit(data)

    conditions = create_conditions(target_name, classes_to_generate)
    generated_data = synt.sample_from_conditions(conditions)

    return generated_data
