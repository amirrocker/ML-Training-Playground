from unittest import TestCase
import numpy as np


from deeqQ_chapter_6.TrainDQNetwork import ExperienceBuffer

class TestExperienceBuffer(TestCase):
	def test_append(self):
		experience_buffer = ExperienceBuffer(100)

		states = np.array([experience] for experience in range(0, 100))

		experience = []
		experience.append()
		self.assertEquals()
		self.fail()

	def test_sample(self):
		self.fail()
