import unittest

from rebel_nerf.uncertainty_nerf.create_noisy_dataset import generate_random


class TestRandomGeneration(unittest.TestCase):
    def test_repeatability(self):
        frames = [
            {"file_path": "test1.png"},
            {"file_path": "test2.png"},
            {"file_path": "test3.png"},
            {"file_path": "test4.png"},
            {"file_path": "test5.png"},
            {"file_path": "test6.png"},
            {"file_path": "test7.png"},
            {"file_path": "test8.png"},
            {"file_path": "test9.png"},
            {"file_path": "test10.png"},
        ]
        dataset_percentage = 0.5

        random_frames_intensity = generate_random(dataset_percentage, frames, seed=0)

        for _ in range(10):
            random_frames_intensity2 = generate_random(
                dataset_percentage, frames, seed=0
            )
            self.assertDictEqual(random_frames_intensity, random_frames_intensity2)
            self.assertEqual(len(random_frames_intensity), 5)

        dataset_percentage = 1
        random_frames_intensity = generate_random(dataset_percentage, frames, seed=0)
        for _ in range(10):
            random_frames_intensity2 = generate_random(
                dataset_percentage, frames, seed=0
            )
            self.assertDictEqual(random_frames_intensity, random_frames_intensity2)
            self.assertEqual(len(random_frames_intensity), 10)

        # change of seed
        random_frames_intensity = generate_random(dataset_percentage, frames, seed=0)
        for _ in range(10):
            random_frames_intensity2 = generate_random(
                dataset_percentage, frames, seed=1
            )
            self.assertNotEqual(random_frames_intensity, random_frames_intensity2)
            self.assertEqual(len(random_frames_intensity), 10)
