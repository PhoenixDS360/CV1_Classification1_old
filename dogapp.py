import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

import streamlit as st
import plotly.express as px
import os

@st.cache(allow_output_mutation=True)
def load_model(model_dir,model_file):
  model=tf.keras.models.load_model(model_dir+model_file)
  return model

def get_disp_img(x):
    img = Image.open(x)
    
    #display the uploaded image
    width, height = img.size    
    aspect_ratio = height / width
    display_height = 200
    display_width = int(display_height / aspect_ratio)
    display_img = img.resize((display_width,display_height))
    
    return display_img


def data_gen(x):
    img = np.asarray(Image.open(x).resize((350, 350)))
    x_test = np.asarray(img.tolist())    
    x_validate = x_test.reshape(1, 350, 350, 3)
 
    return x_validate
  
def predict_img(img,categories_map,model):
    pred = model.predict(img)
    test_pred_label = np.argmax(pred, axis=1) #Identify the class with the maximum probability
    #st.write(test_pred_label)
    # Populate the probabilities for all classes to a dataframe
    out_cols = [k for k, v in sorted(categories_map.items(), key=lambda x: x[1])]
    pred_df = pd.DataFrame()
    
    pred_df[out_cols] = pred
    pred_df_new = pd.melt(pred_df,value_vars = out_cols)
    pred_df_new.rename(columns={"variable":"probable_breed","value":"probability"},inplace=True)
    pred_df_new = pred_df_new.sort_values(["probability"],ascending=False).head(3)
    return pred_df_new
  
def main():
    #constants
    resize_height = 350
    resize_width  = 350
    model_dir  = './'
    model_file = 'effnetl.hdf5'
    sample_files_dir =  './images/'
    
    #mapping between breed names and its integer counterpart
    categories_map = {'affenpinscher': 24,'afghan_hound': 74,'african_hunting_dog': 13,'airedale': 63,'american_staffordshire_terrier': 68,
     'appenzeller': 46,'australian_terrier': 42,'basenji': 7,'basset': 41,'beagle': 95,'bedlington_terrier': 5,'bernese_mountain_dog': 67,
     'black-and-tan_coonhound': 22,'blenheim_spaniel': 110,'bloodhound': 47,'bluetick': 3,'border_collie': 53,'border_terrier': 33,'borzoi': 6,
     'boston_bull': 0,'bouvier_des_flandres': 115,'boxer': 17,'brabancon_griffon': 80,'briard': 91,'brittany_spaniel': 50,'bull_mastiff': 66,
     'cairn': 23,'cardigan': 70,'chesapeake_bay_retriever': 92,'chihuahua': 58,'chow': 82,'clumber': 72,'cocker_spaniel': 118,'collie': 55,
     'curly-coated_retriever': 106,'dandie_dinmont': 93,'dhole': 31,'dingo': 1,'doberman': 18,'english_foxhound': 87,'english_setter': 27,
     'english_springer': 117,'entlebucher': 54,'eskimo_dog': 78,'flat-coated_retriever': 83,'french_bulldog': 114,'german_shepherd': 39,
     'german_short-haired_pointer': 113,'giant_schnauzer': 29,'golden_retriever': 4,'gordon_setter': 88,'great_dane': 109,'great_pyrenees': 94,
     'greater_swiss_mountain_dog': 40,'groenendael': 30,'ibizan_hound': 26,'irish_setter': 45,'irish_terrier': 37,'irish_water_spaniel': 21,
     'irish_wolfhound': 79,'italian_greyhound': 71,'japanese_spaniel': 103,'keeshond': 102,'kelpie': 51,'kerry_blue_terrier': 98,'komondor': 62,
     'kuvasz': 38,'labrador_retriever': 25,'lakeland_terrier': 16,'leonberg': 64,'lhasa': 69,'malamute': 56,'malinois': 61,'maltese_dog': 11,
     'mexican_hairless': 65,'miniature_pinscher': 77,'miniature_poodle': 104,'miniature_schnauzer': 49,'newfoundland': 90,'norfolk_terrier': 12,
     'norwegian_elkhound': 35,'norwich_terrier': 84,'old_english_sheepdog': 75,'otterhound': 19,'papillon': 52,'pekinese': 2,'pembroke': 108,
     'pomeranian': 105,'pug': 60,'redbone': 15,'rhodesian_ridgeback': 44,'rottweiler': 119,'saint_bernard': 76,'saluki': 59,'samoyed': 48,
     'schipperke': 43,'scotch_terrier': 73,'scottish_deerhound': 8,'sealyham_terrier': 100,'shetland_sheepdog': 9,'shih-tzu': 36,'siberian_husky': 89,
     'silky_terrier': 111,'soft-coated_wheaten_terrier': 85,'staffordshire_bullterrier': 86,'standard_poodle': 101,'standard_schnauzer': 20,
     'sussex_spaniel': 112,'tibetan_mastiff': 116,'tibetan_terrier': 34,'toy_poodle': 32,'toy_terrier': 81,'vizsla': 96,'walker_hound': 10,
     'weimaraner': 28,'welsh_springer_spaniel': 57,'west_highland_white_terrier': 97,'whippet': 99,'wire-haired_fox_terrier': 14,'yorkshire_terrier': 107}
    
    categories_reversemap = {ind:breed for breed,ind in categories_map.items()}    
    
    #initial page load
    st.sidebar.header("Dog Breed Identification System")
    st.sidebar.subheader('Choose what you want to do:')
    option = st.sidebar.selectbox("", ["Sample Data", "Upload Image"])
    
    #load the model in cache
    model=load_model(model_dir,model_file)
    
    #Sample Data Function
    if option == "Sample Data":
        filelist=[]
        for root, dirs, files in os.walk(sample_files_dir):
            for file in files:
                filename=os.path.join(root, file)
                filelist.append(filename)
                
        #st.write(filelist)
        filelist = [file for file in filelist if "Thumbs" not in file]
        
        sel_file = st.radio('Select any file: ',filelist)
        disp_img = get_disp_img(sel_file)
        col1,col2 = st.columns(2) 
        col1.image(disp_img,caption="Selected Image")
        
        pred_start = st.button('Identify breed', disabled=False)
        if pred_start:
            with st.spinner('Identifying...'):
                inp_img  = data_gen(sel_file)
                prediction_df = predict_img(inp_img,categories_map,model)
                prediction_df["percentage"] = prediction_df["probability"]*100
                prediction_df["percentage"] = prediction_df["percentage"].round(decimals=3)
                
                fig = px.bar(prediction_df,y="probable_breed",x="percentage", text_auto=True,
                     color='probable_breed',width=500, height=300,orientation='h',
                     title='Prediction Outcomes')
                
                fig.update_traces(showlegend=False)
                fig.update_xaxes(visible=False,showticklabels=True,showgrid=False)
                fig.update_yaxes(title='',showgrid=False,tickfont=dict(family='Arial', size=15, color='black'))
                fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                                   'paper_bgcolor': 'rgba(0, 0, 0, 0)'
                                  })
                col2.plotly_chart(fig)
                
            
    
    
    #Image Upload Function
    if option == "Upload Image":
       st.header("Upload Image")

       upload_img = st.file_uploader("Upload dog's image (in jpg format) to identify its breed", type=['jpg'])
           
       
       
       if upload_img is not None:
           col1,col2 = st.columns(2) 
           disp_img = get_disp_img(upload_img)
           inp_img  = data_gen(upload_img)
           col1.image(disp_img,caption="Uploaded Image")
           col1.write('File Uploaded Successfully!!')
           
           result = col1.button('Identify breed', disabled=False)
           if result:
               with st.spinner('Identifying...'):
                 prediction_df = predict_img(inp_img,categories_map,model)
                 prediction_df["percentage"] = prediction_df["probability"]*100
                 prediction_df["percentage"] = prediction_df["percentage"].round(decimals=3)
                 #st.dataframe(prediction_df)   
                 
                 prediction_df = prediction_df.sort_values('percentage', ascending=True)
                 fig = px.bar(prediction_df,y="probable_breed",x="percentage", text_auto=True,
                      width=500, height=300,orientation='h',
                      title='Prediction Outcomes')
                 
                 fig.update_traces(showlegend=False)
                 fig.update_xaxes(visible=False,showticklabels=True,showgrid=False)
                 fig.update_yaxes(title='',showgrid=False,tickfont=dict(family='Arial', size=15, color='black'))
                 fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                                    'paper_bgcolor': 'rgba(0, 0, 0, 0)'
                                   })
                 col2.plotly_chart(fig)
                       

      
if __name__ == "__main__":
    main()
    