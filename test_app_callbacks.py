import pandas as pd
import plotly.express as px

from app import update_graph_1, get_branch_based_analytics

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
    print("actual")
    actual_fig = update_graph_1(selected_branch)

    # Compare figure data
    assert expected_fig == actual_fig
