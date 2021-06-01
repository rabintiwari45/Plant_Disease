import os
import sys
import warnings
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input,decode_predictions
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from flask import request
from flask import Flask,render_template



app = Flask(__name__)

class_label = ['Apple Scab Leaf','Apple leaf','Apple rust leaf','Bell_pepper leaf',
               'Bell_pepper leaf spot','Blueberry leaf','Cherry leaf','Corn Gray leaf spot',
               'Corn leaf blight','Corn rust leaf','Peach leaf','Potato leaf early blight',
               'Potato leaf late blight','Raspberry leaf','Soyabean leaf','Squash Powdery mildew leaf',
               'Strawberry leaf','Tomato Early blight leaf','Tomato Septoria leaf spot','Tomato leaf',
               'Tomato leaf bacterial spot','Tomato leaf late blight','Tomato leaf mosaic virus',
               'Tomato leaf yellow virus','Tomato mold leaf','grape leaf','grape leaf black rot'
              ]

treatments = [
                '''
                1. Choose resistant varieties when possible.\n
                
                2. Rake under trees and destroy infected leaves to reduce the number of fungal spores available to start the disease cycle over again next spring.\n
                
                3. Water in the evening or early morning hours (avoid overhead irrigation) to give the leaves time to dry out before infection can occur.\n
                
                4. Spread a 3- to 6-inch layer of compost under trees, keeping it away from the trunk, to cover soil and prevent splash dispersal of the fungal spores.\n
                
                5. Apply nitrogen to leaves that have fallen to the ground in the fall to enhance decomposition of fallen leaves and make them more palatable to earthworms.\n
                
                6. Hand apply a liquid fish solution or 16-16-16 fertilizer to help with the decomposition.\n
                
                7. Shred fallen leaves in the fall with a mower to help speed up decomposition.\n
                
                8. Prune your apple trees to open up branching and allow more air circulation.\n
                ''',
                

                
                '''    Your Plant is healthy. It doesn't need any treatments.\\n
                ''',

                '''
                    1. Choose resistant cultivars when available.\n

                    2. Rake up and dispose of fallen leaves and other debris from under trees.\n

                    3. Remove galls from infected junipers. In severe cases, juniper plants should be removed entirely. Keep in mind however, the fungal spores can travel in the wind a long distance so the infected juniper may not even be in your yard.\n

                    4. Apply preventative, disease-fighting fungicides labeled for use on apples weekly, starting with bud break, to protect trees from spores being released by the juniper host. This occurs only once per year, so additional applications after this springtime spread are not necessary.\n

                    5. On juniper, rust can be controlled by spraying plants with a copper fungicide at least four times between late August and late October.

                    6. Safely treat most fungal and bacterial diseases with Fungonil by Bonide
                ''',

                
                '''    Your Plant is healthy. It doesn't need any treatments.
                ''',

                '''
                    1. Copper sprays can be used for control of leaf spot bacteria, but they may not be effective when used alone continuously. Indeed, continuous copper sprays may lead to the development of resistant strains of leaf spot bacteria and thus decrease the effectiveness of copper sprays.

                    2. Products containing microorganisms can be used to enhance plant growth and reduce the negative effects of diseases. These products may contain plant-growth-promoting rhizobacteria (PGPR) or biological agents

                    3. Seed treatment with hot water is effective in reducing bacterial populations on the surface and inside the seeds. 

                    4. Seed treatment with sodium hypochlorite (for example, Clorox) is effective in reducing bacterial populations on seed surfaces. 
                ''',

                '''
                    Your Plant is healthy. It doesn't need any treatments.
                ''',

                '''
                    Your Plant is healthy. It doesn't need any treatments.
                ''',

                '''
                    1. Gray leaf spot overwinters on corn residue and can be especially problematic in corn planted behind corn.  The fungus does not grow well on non-host residues.  Rotating away from corn may help reduce local levels of inoculum and reduce Gray leaf spot severity in the following corn crop. 

                    2. It is essential that resistant varieties be used in fields with a history of Gray leaf spot

                    3. There are numerous fungicides available for control of Gray leaf spot.  However, there is no guarantee that fungicide applications will result in economic returns, especially if they are applied to highly resistant hybrids in fields with little disease.  For this reason it is important to scout fields for symptoms of Gray leaf spot and apply fungicides only when they are needed. 

                    4.  Gray leaf spot severity is unpredictable and multiple factors should be considered when making the decision to use fungicides to control Gray leaf spot.

                ''',

                '''
                    1. choose corn varieties or hybrids that are resistant or at least have moderate resistance to corn leaf blight.

                    2. When you grow corn, make sure it does not stay wet for long periods of time

                    3. The fungus that causes this infection needs between six and 18 hours of leaf wetness to develop. Plant corn with enough space for airflow and water in the morning so leaves can dry throughout the day.

                    4. Tilling the corn into the soil is one strategy, but with a small garden it may make more sense to just remove and destroy the affected plants

                ''',

                '''
                    1. Fungicides can effectively control common rust if initial applications are made
 
                    2. The best management option is growing corn products with higher levels of tolerance to common rust.

                    3. Late-planted corn is more likely to have higher levels of infection since it will likely have young, more susceptible leaves during the time when spores first arrive from overwintering locations.
 
                    4. Water in the early morning hours — avoiding overhead sprinklers — to give plants time to dry out during the day

                ''',

                '''
                    Your Plant is healthy. It doesn't need any treatments.
                ''',

                '''
                    1. Treatment of early blight includes prevention by planting potato varieties that are resistant to the disease; late maturing are more resistant than early maturing varietie.\n

                    2. Avoid overhead irrigation and allow for sufficient aeration between plants to allow the foliage to dry as quickly as possible.\n

                    3. Keep the potato plants healthy and stress free by providing adequate nutrition and sufficient irrigation, especially later in the growing season after flowering when plants are most susceptible to the disease.\n

                    4. For best control, apply copper-based fungicides early, two weeks before disease normally appears or when weather forecasts predict a long period of wet weather.\n

                ''',

                '''
                    1. Eliminate potato cull piles and all other sources of living tubers and eliminate volunteer potatoes from last season.

                    2. Be aware of the relative susceptibility to late blight of the potato varieties that you are planting. Russet Burbank and Snowden are moderately susceptible; Atlantic, Monona, Norchip, Red Norland, Russet Norkotah, and Yukon Gold are very susceptible.

                    3. Plant certified seed and be aware of the late blight situation in the field from which it was harvested. Check the "North American Certified Seed Potato Health Certificate" provided for each lot.

                    4. Minimize handling of seed tubers; if seed is cut, immediately treat with a mancozeb-containing fungicide.

                    5. Hill potatoes to ensure that young tubers are adequately covered by soil.

                    6. Fertilize and irrigate optimally for the variety.
                ''',

                '''
                    Your Plant is healthy. It doesn't need any treatments.
                ''',

                '''
                    Your Plant is healthy. It doesn't need any treatments.
                ''',

                '''
                    1. Use this recipe to make your own solution—mix one tablespoon of baking soda with a teaspoon of dormant oil and one teaspoon of insecticidal or liquid soap (not detergent) to a gallon of water. Spray on plants every one to two weeks.

                    2. Baking Soda (sodium bicarbonate) -This is possibly the best known of the home-made, organic solutions for powdery mildew. Although studies indicate that baking soda alone is not all that effective, when combined with horticultural grade or dormant oil and liquid soap, efficacy is very good if applied in the early stages or before an outbreak occurs.

                    3. Potassium bicarbonate– Similar to baking soda, this has the unique advantage of actually eliminating powdery mildew once it’s there. Potassium bicarbonate is a contact fungicide which kills the powdery mildew spores quickly. In addition, it’s approved for use in organic growing.

                ''',

                '''
                    Your Plant is healthy. It doesn't need any treatments.
                ''',

                '''
                    1. Tomato plants are used to growing in dry climates, so they are unusually sensitive to water on their leaves, which makes them more prone to fungal infections than many other crops.

                    2. Take every precaution you can to minimize the amount of moisture on your tomato plants. Try to avoid working with or around your plants in wet weather.

                    3. When you remove the weeds and volunteer plants, make sure you destroy them – do not place on your compost pile.

                    4. Fertilize properly to maintain vigorous plant growth. Particularly, do not over-fertilize with potassium and maintain adequate levels of both nitrogen and phosphorus.

                ''',

                '''
                    
                    1. Removing infected leaves. Remove infected leaves immediately, and be sure to wash your hands and pruners thoroughly before working with uninfected plants.
                    
                    2. Consider organic fungicide options. Fungicides containing either copper or potassium bicarbonate will help prevent the spreading of the disease. Begin spraying as soon as the first symptoms appear and follow the label directions for continued management.
                    
                    3. Consider chemical fungicides. While chemical options are not ideal, they may be the only option for controlling advanced infections. One of the least toxic and most effective is chlorothalonil (sold under the names Fungonil and Daconil).
                    
                    4. Start with a clean garden. Dispose of all affected plants. The fungus can over-winter on the debris of diseased plants. It's important to dispose of all the affected plants far away from the garden and the compost pile. Keep in mind that it may have spread to your potatoes and eggplants, too.
                ''',

                '''
                    Your Plant is healthy. It doesn't need any treatments.
                ''',

                '''
                    1. Copper fungicides are the most commonly recommended treatment for bacterial leaf spot.

                    2. Varieties with resistance to bacterial spot are available. There are many varieties of bell pepper and hot pepper with resistance to bacterial spot. A few tomato varieties with resistance are available.

                    3. Hot water treatment can be used to kill bacteria on and in seed.

                    4. Avoid high-pressure sprays, as these may injure leaves enough to encourage the introduction of the bacterial pathogen.

                ''',

                '''
                    1. Fertilize and irrigate optimally for the variety.

                    2. Plant resistant cultivars when available.

                    3. Remove volunteers from the garden prior to planting and space plants far enough apart to allow for plenty of air circulation.

                    4. Water in the early morning hours, or use soaker hoses, to give plants time to dry out during the day — avoid overhead irrigation.

                ''',

                '''
                    1. There are no cures for viral diseases such as mosaic once a plant is infected. As a result, every effort should be made to prevent the disease from entering your garden.

                    2. Tomato mosaic virus of tomatoes can exist in the soil or plant debris for up to two years, and can be spread just by touch – a gardener who touches or even brushes up against an infected plant can carry the infection for the rest of the day.  You should wash your hands with soap and disinfect tools after handling tomato plants to keep the disease from spreading.

                    3. Treating mosaic virus is difficult and there are no chemical controls like there are for fungal diseases, although some varieties of tomato are resistant to the disease, and seeds can be bought that are certified disease free.

                ''',

                '''
                    1. Management of TYLCV includes reducing viral inoculum by destroying crop residues.
                    
                    2. using reflective mulches to repel the vector in early stages of crop growth.
                    
                    3. planting TYLCV-resistant varieties when appropriate, and treating plants with a combination of at-plant, drip injected, and foliar insecticides .
                    
                    4. Protection of the crop from TYLCV during the first 5 or 6 wk after transplanting is crucial to reduce yield losses.

                ''',

                '''
                    1. When treating tomato plants with fungicide, be sure to cover all areas of the plant that are above the soil, especially the underside of leaves, where the disease often forms.

                    2.  Calcium chloride-based sprays are recommended for treating leaf mold issues.

                    3. If the tomatoes are being cultivated outdoors, try to keep the leaves dry when watering the plants.

                    4. If possible, water them early in the morning, so that the plants have plenty of time to dry out before the sun comes out.

                ''',

                '''
                    Your Plant is healthy. It doesn't need any treatments.
                ''',

                '''
                    1. Sanitation is extremely important. Destroy mummies, remove diseased tendrils from the wires, and select fruiting canes without lesions.

                    2. Plant grapes in sunny open areas that allow good air movement. If your vines are planted under trees in the shade where they do not get all day sunlight, black rot will be much more difficult to control.

                    3. Using Fungicides to Control Grape Black Rot.

                    4. Shaded areas keep the leaves and fruits from drying and provide excellent conditions for black rot infection and disease development.
                '''

            ]



UPLOAD_FOLDER = "C:/Users/Rabin/Desktop/Image_classification/static"

#MODEL_PATH = 'model.h5'
model = load_model('model.h5')
PATH_TO_SAVED_MODEL = "saved_model"
PATH_TO_LABELS = "leaves_label_map.pbtxt"

def model_predict(img_path,model):
    img = img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    pred = model.predict(img_array)
    pred = np.argmax(pred,axis=1)
    return pred

def object_model_predict(img_path,PATH_TO_SAVED_MODEL,PATH_TO_LABELS):
    detect_fn = tf.compat.v2.saved_model.load(str(PATH_TO_SAVED_MODEL),None)
    category_index = label_map_util.create_category_index_from_labelmap(str(PATH_TO_LABELS),use_display_name=True)
    image_np = plt.imread(img_path)
    input_tensor = tf.convert_to_tensor(image_np, dtype=tf.uint8)
    #input_tensor = tf.convert_to_tensor(image_np)
    print(input_tensor.shape)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    print(input_tensor.shape)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.4,
          agnostic_mode=False)  

    plt.figure()
    plt.figure()
    plt.imshow(image_np_with_detections)
    plt.show()
    print(image_np_with_detections.shape)
    #cv2.imshow(image_np_with_detections)
    return image_np_with_detections #plt.imshow(image_np_with_detections)

@app.route('/')
def home():
   return render_template('index.html')


@app.route('/',methods=['GET','POST'])

def DetectImage():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            preds = model_predict(image_location,model)
            preds_object = object_model_predict(image_location,PATH_TO_SAVED_MODEL,PATH_TO_LABELS)
            #im = Image.fromarray(preds_object)
            im = Image.fromarray((preds_object * 255).astype(np.uint8))
            im.save('static/upload/' + image_file.filename)
            print(preds_object) 

            return render_template("index.html",prediction=class_label[int(preds)],treatment=treatments[int(preds)],image_loc=image_file.filename)
            #return render_template("index.html",image_loc=image_file.filename)

    return render_template('index.html',prediction=None,image_loc=None)





if __name__ == '__main__':
    app.run(port = 2000,debug=True)
