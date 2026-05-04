import os
import tensorflow as tf

def save_data(df, dirname,filename):
    os.makedirs(dirname, exist_ok=True)
    df.to_csv(f'{dirname}/{filename}', index=False)

def save_vocab(tokenizer, filename):
    vocab_dir = f'experiment/vocab/{filename}'
    os.makedirs(vocab_dir,exist_ok = True)
    tokenizer.save_assets(vocab_dir)  #saving the new vocabulary created by the tokenizer in the vocab directory



#this function converts the dataframe to tensorflow dataset, 
def df_to_tfdata(df, topic_lookup, title_tokenizer, batch_size=32, buffer_size=1000, shuffle=False):
    # Extract the news titles and topics
    titles = df['title']    
    labels = df['topic']

    # Converting the titles and topics to integers using the tokenizer
    sequences = title_tokenizer(titles)
    labels = topic_lookup(labels)

    # Combining the numeric representations to a tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((sequences,labels))

    # Shuffling and creating batches of the dataset which will be used for training the model using training dataset
    if shuffle:
        tf_dataset = dataset.shuffle(buffer_size).batch(batch_size)
    else:
        tf_dataset = dataset.batch(batch_size)

    return tf_dataset