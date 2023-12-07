from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Embedding

def load_model_trained() -> Sequential:
    # Create the model
    model = Sequential()
    model.add(Embedding(5332, 50, input_length = 45))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(5332, activation="softmax"))
    model.load_weights('nextword1.h5')
    return model

def load_tokenizer():
    tokenizer = pickle.load(open('my_tokenizer', 'rb'))
    return tokenizer

def correct_word(input_text):
    dict_replace = {'tol': 'tôn', 'toll': 'tôn', 'tole': 'tôn', 'tone': 'tôn', 'tote': 'tôn', 'to le': 'tôn', 'lote': 'tôn', 'hd': 'hộp đèn', 'hđ': 'hộp đèn','told': 'tôn', 'hiflex': 'hiflex', 'hifflex':'hiflex','lightbox': 'hộp đèn', 'side': 'mặt', 'lottone': 'lót tôn', 'alumi': 'aluminium', 'choều':  'chiều',
                    'lo go': 'logo', 'in door': 'indoor', 'helfex': '', 'hefflex': '', 'heflix': '', 'hefflix': '', 'hifflet': 'hiflex', 'hiflet': 'hiflex', 'hjlex': 'hiflex', 'bạtvs': 'bạt', 'điênh': 'điện', 'nhựapvc': 'nhựa pvc', 'lightbo': 'lightbox', 'hixlef': 'hiflex', 'mặ5': 'mặt', 
                    'lop': 'lót', 'lộp': 'lót', 'lốp': 'lót', 'gắng': 'gắn', 'cp': 'chi phí', 'mặc': 'mặt', 'mac': 'mặt', 'mạt': 'mặt', 'sides': 'mặt', '1mặt': 'một mặt', 'out door': 'outdoor', 'đuện': 'điện', 'nguông': 'nguồn', 'choều': 'chiều', 'mặt_': 'mặt', 'đỡl': 'đỡ', 'totle': 'tôn',
                    '1mặc': 'một mặt', '1mac': 'một mặt', '1mạt': 'một mặt', '1măt': 'một mặt','2mặt': 'hai mặt', '2mặc': 'hai mặt', '2mac': 'hai mặt', '2mạt': 'hai mặt', 'uotdoor': 'outdoor', 'chử': 'chữ', 'viềng': 'viền', 'tờ_cán': 'tờ cán', 'kim_ruột': 'kim ruột', 'lótbtole': 'lót tôn',
                    '2măt': 'hai mặt' ,'bản': 'bảng', '1side': 'một mặt', '1sides': 'một mặt', '2side': 'hai mặt', '2sides': 'hai mặt', 'lote': 'tôn', 'top': 'tôn', 'shopname': 'shop name', 'vs': '', 'nỗi': 'nổi', 'manhg': 'mạng', 'caps': 'cáp', 'cphi': 'chi phí', '_bia': 'bia', 'thên': 'thêm',
                    'h.': 'chiều ngang', 'v.': 'chiều dọc', 'vc': 'vận chuyển', 'bh': 'bảng hiệu', 'tolet': 'tôn', 'bạc': 'bạt', 'lsp81': 'lắp', 'c b': 'cb', 'alu': 'aluminium', 'gpm': 'giấy phép', 'gpqc': 'giấy phép', 'phíthu': 'phí thu', 'dọc15m': 'dọc 15m', '1logo': '1 logo', 'role': 'tôn',
                    'mice': 'mica', 'bản': 'bảng', 'bass': 'pát', 'bas': 'pát', 'pass': 'pát', 'bát': 'pát', 'dunghiflex': 'đứng hiflex', 'samkitchen': 'sam kitchen', 'sidehiflet': 'side hiflex', 'kgoong': 'không', 'nguồn12v': 'nguồn 12v', 'lightbx': 'loghtbox', 'diểmv': 'điểm', '1mawjt': '1 mặt', 'tươbgf': 'tường',
                    'thayhiflex': 'thay hiflex', 'bảngă': 'bảng', 'fdieenj': 'điện', 'tolt': 'tôn', 'fides': 'sides', 'ôptole': 'ốp tôn', 'shopnam': 'shopname', 'higlex': 'hiflex', 'cólogo': 'có logo', 'hôph': 'hộp'}
    
    if input_text in dict_replace:
        predicted_word = dict_replace[input_text]
        return predicted_word

def get_prediction_eos(model, tokenizer, input_text):

    token_list = tokenizer.texts_to_sequences([input_text])[0]
    token_list = pad_sequences([token_list], maxlen=45, padding='pre')
    preds = model.predict(token_list)
        #Find the word corresponding to the predicted index in tokenizer.word_index
    predicted_word_index = np.argmax(preds)
    predicted_word = [word for word, index in tokenizer.word_index.items() if index == predicted_word_index][0]
    # Get the indices of the top 5 predicted words
    top5_indices = np.argsort(preds[0])[-5:][::-1]
    # Find the words corresponding to the top 5 indices in tokenizer.word_index
    top5_words = [word for word, index in tokenizer.word_index.items() if index in top5_indices]
    return [predicted_word, top5_words]
