import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os, sys


def output_to_captions(output: np.array, tokenizer) -> list:
    """ Takes the model output (integer encoded) and converts to clean captions """
    def clean(caption: str):
        x = caption.split(" ")
        x = [i for i in x if i != '<pad>' and i != '<end>']
        return " ".join(x)
    captions = tokenizer.sequences_to_texts(output)
    captions = [[clean(i)] for i in captions]
    return captions

def compute_guse(captions):
    def get_google_encoder(sem_dir):
        """ Load/download the GUSE embedding model """
        export_module_dir = os.path.join(sem_dir, "google_encoding")
        if not os.path.exists(export_module_dir):
            module_url = \
                "https://tfhub.dev/google/universal-sentence-encoder/4"
            model = hub.load(module_url)
            tf.saved_model.save(model, export_module_dir)
            print(f"module {module_url} loaded")
        else:
            model = hub.load(export_module_dir)
            print(f"module {export_module_dir} loaded")
        return model

    def get_GUSE_embeddings(sem_dir, sentences, model=None):
        """ Return the GUSE embedding vector """
        if model is None:
            model = get_google_encoder(sem_dir)
        embeddings = model(sentences)
        return embeddings

    guse_model_path = "/home/hpcgies1/Masters-Thesis/AttemptFour/GUSE/GUSE_mode"
    guse_model = get_google_encoder(guse_model_path)
    guse = np.array([np.array(get_GUSE_embeddings(guse_model_path, x, guse_model)) for x in captions])
    print("guse:", guse.shape)
    return guse


if __name__ == '__main__':
    home_dir = "/home/hpcgies1/Masters-Thesis/AttemptFour/"
    model = 'subject_2_baseline2'

    # Load tokenizer
    with open(f"{home_dir}/Log/{model}/eval_out/tokenizer.json", "r") as f:
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())
    print("Tokenizer loaded ...")

    # Load model output
    output = np.squeeze( np.load(open(f"{home_dir}/Log/{model}/eval_out/output_captions_80.npy", "rb")), axis=-1 )

    # Compute guse
    guse = compute_guse(output_to_captions(output, tokenizer))

    # Save output
    np.save(open(f"{home_dir}/Log/{model}/eval_out/output_guse.npy", "wb"), guse)
    print(f"guse saved to disk:\t{model}/eval_out/")
