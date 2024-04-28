import pandas as pd
import plotly.express as px
import numpy as np

from app import update_graph_1, get_branch_based_analytics, update_graph_2, update_output

# Test 'AIML' Branch domain analysis
def test_update_graph_branchwise_domain_analysis():
    selected_branch = 'AIML'

    expected_data = {
        'Domain': ['Machine Learning', 'Web Development', 'OpenCV', 'NLP', 'App Development', 'Web Scraping', 'IoT'],
        'Count': [61, 26, 16, 12, 6, 3, 3]
    }

    expected_df = pd.DataFrame(expected_data)
    expected_fig = px.bar(
        expected_df,
        x='Count', y='Domain',
        labels={'Domain': 'Domain', 'Count': 'Count'},
        title=f'Analytics for Branch {selected_branch}', orientation='h'
    )

    # Get the actual figure
    actual_fig = update_graph_1(selected_branch)

    # Compare figure data
    assert expected_fig == actual_fig


# Test 'ESE marks' measure of central tendency (mean, median, mode)
def test_measure_of_central_tendency():
    selected_column = 'ESE marks'

    expected_fig = px.bar(x=['Mean', 'Median', 'Mode'], y=[82.93506493506493, 83, 82 ], labels={'x': 'Measure', 'y': selected_column})
    expected_fig.update_layout(title=f'Measure of Central Tendency for {selected_column}')
    # print(expected_fig)

    actual_fig = update_graph_2(selected_column)
    # print(actual_fig)

    # Expected
    expected_x = expected_fig.data[0]['x'] # X axis = ['Mean', 'Median', 'Mode']
    expected_y = expected_fig.data[0]['y'] # Values = [82.93506493506493, 83, 82]

    actual_x = actual_fig.data[0]['x']
    actual_y = actual_fig.data[0]['y']

    assert np.array_equal(expected_x, actual_x)
    assert np.array_equal(expected_y, actual_y)


# Test for faculty recommendation for 'Machine Learning'
def test_faculty_recommendation():
    subject = 'MacHinE LEarniNg'
    expected_professors = [['Prof. Jignesh Sisodia', 0.9821875000000001], ['Prof. Swapnali Kurhade',0.922684733265651], ['Prof. Khushbhu Chauhan', 0.7680410147421811], ['Dr. Surekha Dholay',  0.7590451769533382], ['Prof. Rupali sawant', 0.7484931568147637]] 

    actual_professors = update_output(1, subject)
    actual_professors = actual_professors.reset_index()
    actual_professors = actual_professors.values.tolist()
    
    assert expected_professors == actual_professors
