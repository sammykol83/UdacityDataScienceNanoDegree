import pandas as pd
import plotly.graph_objs as go
from sqlalchemy import create_engine

def return_figures():
    """Creates four plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the four plotly visualizations

    """

    # load data
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df = pd.read_sql_table('InsertTableName', engine)

    # Extract data for 1st graph
    words_per_sms_list = []
    for sentence in df['message']:
        words_per_sms_list.append(len(sentence.split()))

    graph_one = []
    graph_one.append(go.Histogram(x=words_per_sms_list, xbins=dict(start=1,end=200,size=1), histnorm='probability'))
    layout_one = dict(title='Number of words per SMS',
                      xaxis=dict(title='#words'),
                      yaxis=dict(title='Probability'),
                      )

    # second chart Gives us a "peeking" table to see how the messages look
    graph_two = []

    graph_two.append(
        go.Table(
            header=dict(values=['Raw Messages'],
                        fill=dict(color='#C2D4FF')),
            cells=dict(values=[df.message],
                       fill=dict(color='#F5F8FF'))
        )
    )

    layout_two = dict(title='Data peek',
                       xaxis=dict(title='Values'),
                       yaxis=dict(title='Columns'),
                       )

    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    return figures
