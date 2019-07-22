import pandas as pd
import plotly.graph_objs as go

# Use this file to read in your data and prepare the plotly visualizations. The path to the data files are in
# `data/file_name.csv`

def return_figures():
    """Creates four plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the four plotly visualizations

    """

    # Load the data
    df = pd.read_csv(r'data\googleplaystore.csv')

    # Drop null values. I don't care about accuracy or imputation etc.. (for this exercise).
    # Approximately 10% of rows dropeed.
    df_clean = df.dropna()

    # First plot will be the 5 most common category types of apps on google play
    cat_counts = df_clean['Category'].value_counts()[0:5]
    x = cat_counts.index.tolist()
    y = (cat_counts.values/df_clean.shape[0]*100).tolist()
    # first chart plots arable land from 1990 to 2015 in top 10 economies 
    # as a line chart
    
    graph_one = []    
    graph_one.append(go.Bar(x=x, y=y))
    layout_one = dict(title='Most common google play categories',
                    xaxis=dict(title = 'categories'),
                    yaxis=dict(title = 'Percent of total aps'),
                )

    # second chart plots the distribution of android versions for app store
    # Obviously here we would need some data wrangling because definitions of versions are overlapping!
    graph_two = []
    x = df_clean['Android Ver'].value_counts().index.tolist()[0:5]
    y = (df_clean['Android Ver'].value_counts()).tolist()[0:5]
    #graph_two.append(go.Bar(x=x, y=y))
    graph_two.append(go.Pie(labels=x, values=y))

    layout_two = dict(title = 'Distribution of Android versions(5 most common)',
                xaxis = dict(title = 'Version',),
                yaxis = dict(title = 'Percent of total aps'),
                )


# third chart plots percent of population that is rural from 1990 to 2015
    graph_three = []

    # Wrangle data a bit.. We want to sort installations and find average price per number of installations
    prices = df_clean['Price']
    installs = df_clean['Installs']

    # Remove '$' sign for prices
    new_prices = []
    for price in prices:
        new_prices.append(float(price.replace('$', '')))

    # Remove '+' and ',' from number of installations
    new_installs = []
    for install in installs:
        tmp = install.replace(',', '')
        new_installs.append(float(tmp.replace('+', '')))

    new_column_1 = pd.to_numeric(pd.Series(new_prices, name='Price', index=df_clean['Price'].index.values))
    df_clean.update(new_column_1)

    new_column_1 = pd.to_numeric(pd.Series(new_installs, name='Installs', index=df_clean['Installs'].index.values))
    df_clean.update(new_column_1)

    # Filter only paid apps
    df_tmp = df_clean[df_clean['Price'] > 0]

    # Group paid apps by number of installs
    grouped = df_tmp.groupby('Installs')

    installs_list = []
    avg_price_list = []
    for name, group in grouped:
        installs_list.append(name)
        avg_price_list.append(group['Price'].mean())

    graph_three.append(go.Scattergl(y=avg_price_list, x=installs_list))
    layout_three = dict(title = 'Price vs. Installs', xaxis = dict(title = 'Installs'),
                        yaxis = dict(title = 'Price(USD)'))
    
# fourth chart shows rural population vs arable land
    graph_four = []

    graph_four.append(
        go.Table(
            header=dict(values=list(df_clean.columns[4:8]),
                        fill=dict(color='#C2D4FF'),
                        align=['left'] * 5),
            cells=dict(values=[df_clean.Size, df_clean.Installs, df_clean.Type, df_clean.Price],
                       fill=dict(color='#F5F8FF'),
                       align=['left'] * 5))
      )

    layout_four = dict(title = 'Data peek',
                xaxis = dict(title = 'Values'),
                yaxis = dict(title = 'Columns'),
                )
    
    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    figures.append(dict(data=graph_four, layout=layout_four))

    return figures