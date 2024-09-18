# test_visualization.py

import unittest
import pandas as pd
from visualization import VisualizationHelper, DataVisualizer, AgentVisualizer

class TestVisualizationHelper(unittest.TestCase):
    def setUp(self):
        self.helper = VisualizationHelper()
        self.test_data = {
            "Category": ["A", "B", "C"],
            "Values": [10, 15, 7]
        }
        self.df = pd.DataFrame(self.test_data)

    def test_load_data(self):
        self.helper.data = self.df
        self.assertTrue(self.helper.data.equals(self.df), "Data should be loaded correctly")

    def test_clean_data(self):
        self.helper.data = pd.DataFrame({
            "Category": ["A", "B", None],
            "Values": [10, None, 7]
        })
        cleaned_data = self.helper.clean_data()
        self.assertNotIn(None, cleaned_data['Category'], "Data should not contain None values after cleaning")
        self.assertNotIn(None, cleaned_data['Values'], "Data should not contain None values after cleaning")

    def test_create_bar_plot(self):
        self.helper.data = self.df
        fig = self.helper.create_bar_plot(x_col="Category", y_col="Values", title="Test Bar Plot")
        self.assertEqual(fig.layout.title.text, "Test Bar Plot", "Plot title should match")

class TestDataVisualizer(unittest.TestCase):
    def setUp(self):
        self.visualizer = DataVisualizer(filepath="dummy.csv")
        self.visualizer.helper.data = pd.DataFrame({
            "Category": ["A", "B", "C"],
            "Values": [10, 15, 7]
        })

    def test_load_and_clean_data(self):
        with unittest.mock.patch.object(VisualizationHelper, 'load_data', return_value=self.visualizer.helper.data):
            self.visualizer.load_and_clean_data()
            self.assertIsNotNone(self.visualizer.data, "Data should be loaded and cleaned")

    def test_generate_plot(self):
        fig = self.visualizer.generate_plot()
        self.assertIsNotNone(fig, "Plot should be generated")

class TestAgentVisualizer(unittest.TestCase):
    def setUp(self):
        self.agent_visualizer = AgentVisualizer()

    def test_set_data(self):
        new_data = {
            "Agent": ["Agent1", "Agent2", "Agent3"],
            "Performance": [75, 85, 95]
        }
        self.agent_visualizer.set_data(new_data)
        df = self.agent_visualizer.get_data()
        self.assertTrue((df['Performance'] == [75, 85, 95]).all(), "Performance values should match the new data")

    def test_generate_plot(self):
        fig = self.agent_visualizer.generate_plot()
        self.assertIsNotNone(fig, "Plot should be generated")

if __name__ == '__main__':
    unittest.main()
