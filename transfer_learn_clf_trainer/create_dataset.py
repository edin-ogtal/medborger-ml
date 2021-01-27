from __future__ import absolute_import, division, print_function

import csv
import os
import datasets

class MedborgerDataset(datasets.GeneratorBasedBuilder):
    """Medborger dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=["Angreb", "Anerkendelse"]),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(os.path.join(os.environ["SM_CHANNEL_DATA"], 'train.csv'))
        eval_path = dl_manager.download_and_extract(os.path.join(os.environ["SM_CHANNEL_DATA"],'valid.csv'))
        test_path = dl_manager.download_and_extract(os.path.join(os.environ["SM_CHANNEL_DATA"],'test.csv'))
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": eval_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', delimiter="\t", quoting=csv.QUOTE_ALL, skipinitialspace=True
            )
            for id_, row in enumerate(csv_reader):
                if id_ == 0:
                    pass
                else:
                    label, text = row
                    label = int(label) - 1                
                    yield id_, {"text": text, "label": label}