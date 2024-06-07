import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

intent_labels = [
    'Complain', 'Praise', 'Apologize', 'Thank', 'Criticize',
    'Agree', 'Taunt', 'Flaunt','Joke', 'Oppose',
    'Comfort', 'Care', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave',
    'Prevent', 'Greet', 'Ask for help']
def viewConfusin(confusin_df):

    # lows=confusin_df.shape[0]
    # for i in range(lows):
    #     confusin_df.iloc[i] = confusin_df.iloc[i] / confusin_df.iloc[i].sum()

    for col in confusin_df.columns:
        confusin_df[col]=confusin_df[col] / confusin_df[col].sum()


    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusin_df.to_numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + intent_labels, rotation=60)
    ax.set_yticklabels([''] + intent_labels,rotation=0)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig('..\\results\\nmfir_bert_confusion2.png',dpi=600)
    plt.show()

if __name__ == '__main__':
    confusin_df=pd.read_csv('..\\results\\nmfir_bert_confusion2.csv')
    viewConfusin(confusin_df)
    


