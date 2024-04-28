
from dash import Dash, html,dash_table,dcc,Input,Output,callback
import numpy as np
import pandas as pd
import sqlite3
import plotly.express as px
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


#tailwind config
external_script = ["https://tailwindcss.com/", {"src": "https://cdn.tailwindcss.com"}]

# Connect to the SQLite database
conn = sqlite3.connect("project.db")

# Read the entire 'projects' table into a DataFrame
df = pd.read_sql("SELECT * FROM projects", conn)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

df['Project Title'] = df['Project Title'].apply(remove_stopwords)

for col in ["Field/Market Survey & Literature Survey (10)","Problem Definition (5)","Presentation (10)","Project Planning (5)","Phase 1 marks","Architecture,Design/Algorithm Testing/Simulation/ System Flow Diagram (DFD)","70% Project Implementaion (20)","Ethics","Team Work (5)","Presentation(5)","Phase 2 marks","Presentation (10)","Implementation","Project Report (10)","Phase 3 marks","ESE marks"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

#first three boxes
total_project_count = df['Project Title'].nunique()
total_domain_count = df['Domain'].str.split(',').explode().nunique()
projects_above_85 = df[df['ESE marks'] > 85]
# Count the number of projects
num_projects_above_85 = len(projects_above_85)


#domain wise analytics
#overall analytics
domains = df['Domain'].str.split(',').explode().str.strip()
domains_counts = domains.value_counts()
pie = px.pie( values=domains_counts.values,names=domains_counts.index,title='Overall Distribution of Domains')

#branch wise analytics
def get_branch(uid):
    fifth_digit = str(uid)[4]
    if fifth_digit == '3':
        return 'COMPS'
    elif fifth_digit == '6':
        return 'AIML'
    else:
        return 'DS'
    
df['Branch'] = df['UID'].apply(get_branch)

def get_branch_based_analytics(branch_name):
    branch_df = df[df['Branch'] == branch_name]
    branch_df = branch_df['Domain'].str.split(',').explode().str.strip().reset_index(drop=True)  # Reset index
    branch_domain_counts = branch_df.value_counts().reset_index()  # Convert Series to DataFrame
    branch_domain_counts.columns = ['Domain', 'Count']  # Rename columns
    branch_domain_counts = branch_domain_counts.sort_values(by='Count', ascending=False)  
    return branch_domain_counts

@callback(
    Output('branch-domain-pie', 'figure'),
    [Input('branch-domain-radio', 'value')]
)
def update_graph_1(selected_branch):
    branch_domain_counts = get_branch_based_analytics(selected_branch)
    fig = px.bar(branch_domain_counts, y='Domain',x='Count',
                 labels={'Domain': 'Domain', 'Count': 'Count'}, title=f'Analytics for Branch {selected_branch}',orientation='h')
    return fig

# Filter data based on selected domain
df['ESE marks'] = pd.to_numeric(df['ESE marks'], errors='coerce')

domain_df = df.assign(Domain=df['Domain'].str.split(', ')).explode('Domain')

# Calculate the weighted average ESE marks for each domain based on the number of projects
weighted_avg_ese = (domain_df.groupby('Domain')['ESE marks']
                    .mean()  # Calculate the mean ESE marks for each domain
                    .reset_index(name='Mean ESE marks'))

# Create a bar chart
domain_based_average_bar = px.bar(weighted_avg_ese, x='Domain', y='Mean ESE marks', 
                labels={'Domain': 'Project Domain', 'Mean ESE marks': 'Weighted Average ESE Marks'},
                title='Weighted Average ESE Marks by Project Domain')

#find unexplored domains
df1 = pd.read_csv('./project_guide.csv')
faculty_domains = []
for expertise in df1['Domain Expertise']:
    faculty_domains.extend(expertise.split(','))
faculty_domains = [domain.strip() for domain in faculty_domains]
faculty_domains = list((faculty_domains))
student_domains = []
for domain_list in df['Domain']:
    student_domains.extend(domain_list.split(','))
student_domains = [domain.strip() for domain in student_domains]
student_domains = list((student_domains))
unexplored_domains = [domain for domain in faculty_domains if domain not in student_domains]
unexplored_domains_data = {}
total_domains = len(faculty_domains)
for domain in set(faculty_domains):
    d = faculty_domains.count(domain)
    if domain not in student_domains:
        percentage = (d / total_domains) * 100
        unexplored_domains_data[domain] = percentage
unexplored_domains_df = pd.DataFrame(list(unexplored_domains_data.items()), columns=['Domain', 'Percentage'])
unexplored_domains_df['Percentage'] = pd.to_numeric(unexplored_domains_df['Percentage'], errors='coerce')
unexplored_domains_df = unexplored_domains_df.dropna()
unexplored_domains_fig = px.sunburst(unexplored_domains_df, path=['Domain'], values='Percentage',
                    title='Percentage of Unexplored Domains by Students')

@callback(
    Output('wordcloud-graph', 'figure'),
    [Input('wordcloud-graph', 'id')]
)
def update_wordcloud(_):
    # Combine all project titles into a single string
    text = ' '.join(df['Project Title'])

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Plot word cloud
    fig = px.imshow(wordcloud)
    fig.update_layout(title='Word Cloud of Project Titles')

    return fig

@callback(
    Output('central-tendency-graph', 'figure'),
    [Input('column-dropdown', 'value')])
def update_graph_2(selected_column):
    # Calculate the measure of central tendency
    mean = df[selected_column].mean()
    median = df[selected_column].median()
    mode = df[selected_column].mode()[0]

    # Create the bar chart
    fig = px.bar(x=['Mean', 'Median', 'Mode'], y=[mean, median, mode], labels={'x': 'Measure', 'y': selected_column})
    fig.update_layout(title=f'Measure of Central Tendency for {selected_column}')

    return fig


#heatmap plot

domain_marks = df[['Domain', '70% Project Implementaion (20)']].assign(Domains=df['Domain'].str.split(', ')).explode('Domains')

@callback(
    Output('marks-heatmap', 'figure'),
    [Input('column-dropdown', 'value')])
def update_graph_3(selected_column):
    # Create the scatter plot
    marks_columns = ["Field/Market Survey & Literature Survey (10)","Problem Definition (5)","Presentation (10)","Project Planning (5)","Phase 1 marks","Architecture,Design/Algorithm Testing/Simulation/ System Flow Diagram (DFD)","70% Project Implementaion (20)","Ethics","Team Work (5)","Presentation(5)","Phase 2 marks","Presentation (10)","Implementation","Project Report (10)","Phase 3 marks","ESE marks"]
    short_marks_columns = [col[:5] for col in marks_columns]

    fig = px.imshow(df[marks_columns].corr(), x=short_marks_columns, y=short_marks_columns,
                labels=dict(color="Correlation"),
                title="Correlation Heatmap of Marks Columns")
    fig.update_layout(title='Heatmap for Marks Columns')

    return fig

@callback(
    Output('stacked-bar-chart', 'figure'),
    [Input('column-dropdown', 'value')])
def update_graph_4(selected_column):
    phases = ["Phase 1 marks", "Phase 2 marks", "Phase 3 marks"]
    categories_per_phase = {
        "Phase 1 marks": ["Field/Market Survey & Literature Survey (10)", "Problem Definition (5)",
                        "Presentation (10)", "Project Planning (5)"],
        "Phase 2 marks": ["Architecture,Design/Algorithm Testing/Simulation/ System Flow Diagram (DFD)",
                        "70% Project Implementaion (20)", "Ethics", "Team Work (5)", "Presentation(5)"],
        "Phase 3 marks": ["Presentation (10)", "Implementation", "Project Report (10)"]
    }

    # Calculate the average for each phase and each category
    averages = {}
    for phase in phases:
        phase_categories = categories_per_phase[phase]
        phase_data = df[phase_categories]
        averages[phase] = phase_data.mean()

    # Convert the averages to a dataframe
    averages_df = pd.DataFrame(averages)

    # Melt the dataframe for plotting
    melted_df = averages_df.reset_index().melt(id_vars="index", var_name="Phase", value_name="Average")
    # Create the bar chart
    # Create the bar chart
    fig = px.bar(melted_df, x="Phase", y="Average", color="index",
                 title="Average Marks Across Phases",
                 labels={"Average": "Average Marks", "index": "Category", "Phase": "Phase"},
                 category_orders={"Phase": phases},
                 hover_data={"index": True, "Average": True})  # Include full labels in hover
    
    # Set the x-axis tickmode to "array" and provide the labels
    fig.update_xaxes(tickmode="array", tickvals=melted_df["Phase"].unique(), ticktext=melted_df["Phase"].unique())
    
    # Truncate labels and set hovertemplate
    for trace in fig.data:
        trace.customdata = np.full((len(trace.y), 1), trace.name)
        trace.name = trace.name[:10] + '...' if len(trace.name) > 10 else trace.name
    
    # Show the chart
    return fig

# recommendation system


student_df = pd.read_csv("./data.csv")
faculty_df = pd.read_csv("./project_guide.csv")

# Preprocess the data
faculty_df['Domain Expertise'] = faculty_df['Domain Expertise'].str.replace(',', ' ')
student_df['ESE marks'] = pd.to_numeric(student_df['ESE marks'], errors='coerce')

# Calculate the average ESE marks for each faculty
faculty_ese = student_df.groupby('Project Guide')['ESE marks'].mean()

# Create a scoring system
vectorizer = CountVectorizer()
faculty_vectors = vectorizer.fit_transform(faculty_df['Domain Expertise'])
cosine_sim = cosine_similarity(faculty_vectors)

@callback(
    Output('output-container-button', 'children'),
    [Input('submit-val', 'n_clicks')],
    [Input('user_domains', 'value')]
)
def update_output(n_clicks, value):
    if n_clicks > 0:
        # Get the domains that the user is interested in
        user_domains = [value]

        # Transform the user_domains list into a vector
        user_vector = vectorizer.transform([' '.join(user_domains)])

        # Calculate the cosine similarity between the user_vector and the faculty vectors
        user_sim = cosine_similarity(faculty_vectors, user_vector)

        # Flatten the user_sim array and create a Series with the faculty names as the index
        domain_scores = pd.Series(user_sim.flatten(), index=faculty_df['Name of Faculty'])

        # Normalize the domain scores
        scaler = MinMaxScaler()
        domain_scores = pd.Series(scaler.fit_transform(domain_scores.values.reshape(-1, 1)).flatten(), index=faculty_df['Name of Faculty'])

        # Recalculate faculty_ese within the callback function
        faculty_ese = student_df.groupby('Project Guide')['ESE marks'].mean()
        faculty_ese = pd.Series(scaler.fit_transform(faculty_ese.values.reshape(-1, 1)).flatten(), index=faculty_ese.index)

        # Calculate the final scores with 80% weight to domain expertise and 20% weight to ESE marks
        scores = 0.8 * domain_scores + 0.2 * faculty_ese

        # Get the top 5 faculty recommendations
        top_5_recommendations = scores.sort_values(ascending=False).head(5)

        return html.Div([
            html.H4('Top 5 Faculty Recommendations:', className="text-lg font-bold mt-3"),
            html.Ul([html.Li(faculty, className="border border-gray-300 p-2 rounded mb-2") for faculty in top_5_recommendations.index], className="divide-y divide-gray-300")
        ])




# Create a Dash app
app = Dash(__name__,external_scripts=external_script)

app.layout = html.Div(
    className='bg-[#f1f1fb] min-h-screen flex flex-col',
    children=[
        html.Div(
            className='text-3xl p-4 font-semibold ',
            children='Dashboard for Mini Project Data Analysis'
        ),
        html.Div(
            className='flex justify-around ',
            children=[
                html.Div(
                    className='w-[30%] h-[25vh]  rounded-lg flex flex-col items-center p-4 bg-white',
                    children=[
                        html.Div( className='text-xl h-[30%] flex items-center  font-semibold',children='Total number of projects created: '),
                        html.Div(className='text-6xl h-[70%]  flex items-center',children=total_project_count)
                    ]
                ),
                html.Div(
                    className='w-[30%] h-[25vh]  rounded-lg flex flex-col items-center p-4 bg-white',
                    children=[
                        html.Div( className='text-xl h-[30%] flex items-center  font-semibold',children='Number of Unique Domains Explored: '),
                        html.Div(className='text-6xl h-[70%]  flex items-center',children=total_domain_count)
                    ]
                ),
                html.Div(
                    className='w-[30%] h-[25vh]  rounded-lg flex flex-col items-center p-4 bg-white',
                    children=[
                        html.Div( className='text-xl h-[30%] flex items-center  font-semibold',children='Number of Students above 85 marks: '),
                        html.Div(className='text-6xl h-[70%]  flex items-center',children=num_projects_above_85)
                    ]
                ),
            ]
        ),
        html.Div(
            className='p-6 flex flex-col gap-6',
            children=[
                html.Div(className='text-2xl font-bold',children='Domain Wise analysis of Projects:'),
                html.Div(
                    className='flex flex-1 justify-around min-h-[50vh] ',
                    children=[
                        dcc.Graph(className='w-[45%] rounded-lg border-5 border-black ',figure=pie),
                        html.Div(
                            className='w-[45%] bg-white',
                            children=[
                                dcc.RadioItems(
                                    id='branch-domain-radio',
                                    labelStyle={'display': 'inline-block', 'marginTop': '15px','paddingLeft':'2rem'},
                                    options=[{'label': branch, 'value': branch} for branch in df['Branch'].unique()],
                                    value=df['Branch'].unique()[1]  # Default value

                                ),
                                dcc.Graph(id='branch-domain-pie')

                            ]
                        )
                    ]
                ),
                html.Div(
                    className='flex flex-1 justify-around min-h-[50vh] ',
                    children=[
                        html.Div(
                            className='w-[45%] bg-white',
                            children=[
                                dcc.Graph(
                                    figure=domain_based_average_bar
                                )
                            ]
                        ),
                        html.Div(
                            className='w-[45%] bg-white',
                            children=[
                                dcc.Graph(
                                    figure=unexplored_domains_fig
                                )
                            ]
                        )
                    ]
                ),
                
            ]   
        ),
        html.Div(
            className='p-6 flex flex-col gap-6',
            children=[
                html.Div(className='text-2xl font-bold',children='Project Detail analysis:'),
                html.Div(
                    className='flex flex-1 justify-around min-h-[50vh] ',
                    children=[
                        dcc.Graph(id='wordcloud-graph'),
                         html.Div(className='w-[45%] bg-white p-6 ',children=[
                            html.H1(className='pb-4',children='Measure of Central Tendency for Marks Columns'),
                            dcc.Dropdown(
                                id='column-dropdown',
                                options=["Field/Market Survey & Literature Survey (10)","Problem Definition (5)","Presentation (10)","Project Planning (5)","Phase 1 marks","Architecture,Design/Algorithm Testing/Simulation/ System Flow Diagram (DFD)","70% Project Implementaion (20)","Ethics","Team Work (5)","Presentation(5)","Phase 2 marks","Presentation (10)","Implementation","Project Report (10)","Phase 3 marks","ESE marks"],
                                value='Phase 1 marks'
                            ),
                            dcc.Graph(id='central-tendency-graph')
                        ])
                    ]
                ),
                html.Div(
                        className='flex flex-1 justify-around min-h-[50vh] ',
                        children=[
                            html.Div(className='w-[45%] bg-white',children=[
                            dcc.Graph(id='marks-heatmap')
                        ]),
                        html.Div(
                            className='w-[45%] bg-white',
                            children=[
                                dcc.Graph(
                                    id='stacked-bar-chart'
                                )
                            ]
                        )
                    ]
                ),
                
                
            ]   
        ),
        html.Div([
    html.Div([
        html.H1("Faculty Recommendation System", className="text-3xl text-center font-bold mt-8 mb-4"),
        html.Div([
            dcc.Input(id='user_domains', type='text', placeholder='Enter your domains of interest', className="flex-grow border border-gray-300 rounded-md p-2 mr-2",autoComplete='off' ),
            html.Button('Submit', id='submit-val', n_clicks=0, className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"),
        ], className="flex items-center")
    ], className="container mx-auto"),
    html.Div(id='output-container-button', className="container mx-auto mt-3")
])
        
    ]
)
# Run the app
app.css.config.serve_locally = True

if __name__ == '__main__':
    app.run_server(debug=True)