from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RandomHeight, RandomWidth, RandomRotation, RandomZoom, RandomFlip
import os
from werkzeug.utils import secure_filename
import tempfile
import uuid

app = Flask(__name__)

# Load the models
MODEL_DIR = 'models'
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)
model_mapping = {
    'tomato': 'tomato_densenet_finetuned_model.h5',
    'corn': 'corn_densenet_finetuned_model.h5',
    'tea': 'tea_densenet_finetuned_model.h5',
    'cotton': 'cotton_densenet_finetuned_model.h5',
    'potato': 'potato_densenet_finetuned_model.h5',
    'mango': 'mango_densenet_finetuned_model.h5',
    'rice': 'rice_densenet_finetuned_model.h5'
}

_model_cache = {}

def load_model(plant_type):
    # Return cached model if available
    if plant_type in _model_cache:
        return _model_cache[plant_type]

    model_filename = model_mapping.get(plant_type)
    if not model_filename:
        raise ValueError('Invalid plant type selected')

    model_path = os.path.join(MODEL_DIR, model_filename)
    # Register custom objects for data augmentation layers
    custom_objects = {
        'RandomHeight': RandomHeight,
        'RandomWidth': RandomWidth,
        'RandomRotation': RandomRotation,
        'RandomZoom': RandomZoom,
        'RandomFlip': RandomFlip
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    _model_cache[plant_type] = model
    return model

def _is_allowed_file(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return render_template('index.html', prediction='No file part')

            file = request.files['file']
            if not file or file.filename == '':
                return render_template('index.html', prediction='No selected file')

            if not _is_allowed_file(file.filename):
                return render_template('index.html', prediction='Invalid file type. Please upload a JPG or PNG image.')

            plant_type = request.form.get('plant_type', '')
            if plant_type not in model_mapping:
                return render_template('index.html', prediction='Please select a valid plant type.')

            safe_name = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex}_{safe_name}"
            file_path = os.path.join(UPLOAD_DIR, unique_name)
            file.save(file_path)

            model = load_model(plant_type)

            img = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x /= 255.0

            preds = model.predict(x)
            pred_class = np.argmax(preds, axis=1)

            result = get_class(plant_type, pred_class[0])

            return render_template(
                'index.html',
                prediction=result,
                diseases_details=diseases_details.get(plant_type, {}).get(result, {})
            )
        except Exception as exc:
            return render_template('index.html', prediction=f'Error: {str(exc)}')
        finally:
            # Best-effort cleanup of uploaded file
            try:
                if 'file_path' in locals() and os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass

    # GET request
    return render_template('index.html', prediction=None)

def get_class(plant_type, index):
    # Return labels that exactly match the keys used in `diseases_details` so lookups succeed
    classes = {
        'tomato': ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 'Septoria Leaf Spot',
                   'Spider Mites (Two-spotted Spider Mite)', 'Target Spot', 'Yellow Leaf Curl Virus', 'Mosaic Virus', 'Healthy'],
        'corn': ['Northern Corn Leaf Blight', 'Common Rust', 'Gray Leaf Spot', 'Healthy Corn Plants'],
        'tea': ['Anthracnose', 'Algal Leaf Spot (Cephaleuros)', 'Bird Eye Spot', 'Brown Blight', 'Grey Blight', 'Maintaining Healthy Tea Plants', 'Red Leaf Spot', 'White Spot'],
        'cotton': ['Cotton Leaf Curl Virus (CLCuV)', 'Cotton Bacterial Blight', 'Healthy Cotton Plants'],
        'potato': ['Early Blight', 'Late Blight', 'Healthy Potato Plants'],
        'mango': ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Maintaining Healthy Mango Plants', 'Powdery Mildew', 'Sooty Mould'],
        'rice': ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']
    }
    return classes[plant_type][index]




############TOMATO
diseases_details = {
    "tomato": {
        "Bacterial Spot": {
            "disease": "Caused by several species of Xanthomonas, it creates small, water-soaked spots on leaves, stems, and fruits, eventually turning into scab-like lesions.",
            "occurrence": "Spread through water (rain or irrigation) and contaminated tools/equipment. It thrives in warm, moist conditions.",
            "prevention": "Use resistant varieties, apply copper-based sprays early, and practice crop rotation. Avoid working in fields when plants are wet to prevent spread.",
            "cure": "Once established, control is difficult. Copper sprays and improved cultural practices can mitigate but not eliminate the disease."
        },
        "Early Blight": {
            "disease": "Caused by the fungus Alternaria solani, it manifests as dark, concentric rings on older leaves and stems, leading to leaf drop.",
            "occurrence": "The fungus overwinters in the soil and plant debris, infecting through splashing rain or irrigation.",
            "prevention": "Rotate crops, till under infected debris, and practice good weed management. Use fungicide sprays when conditions favor disease development.",
            "cure": "Apply fungicides based on mancozeb or chlorothalonil, especially after periods of rain or heavy dew."
        },
        "Late Blight": {
            "disease": "A serious disease caused by Phytophthora infestans, leading to rapid wilting, brown lesions on leaves, stems, and fruit.",
            "occurrence": "It spreads through spores in moist, cool weather and can devastate crops quickly.",
            "prevention": "Grow resistant varieties, improve air circulation with proper spacing, and avoid overhead irrigation. Use fungicides preventatively in high-risk areas.",
            "cure": "Apply specific fungicides promptly at the sign of outbreak. Infected plants should be removed and destroyed."
        },
        "Leaf Mold": {
            "disease": "Caused by Fulvia fulva (formerly Cladosporium fulvum), resulting in pale green to yellow spots on upper leaf surfaces and a velvety, olive-green to brown mold on undersides.",
            "occurrence": "Favored by high humidity and temperatures between 71-79°F (22-26°C). It's more common in greenhouse tomatoes.",
            "prevention": "Ensure good air circulation, reduce humidity in greenhouses, and use resistant varieties.",
            "cure": "Apply fungicides that target leaf mold specifically. Improved ventilation and reduced leaf wetness can also reduce disease pressure."
        },
        "Septoria Leaf Spot": {
            "disease": "Caused by Septoria lycopersici, this disease features small, circular spots with greyish centers and dark margins on leaves.",
            "occurrence": "The fungus survives in plant debris and soil, spreading through splashing water.",
            "prevention": "Practice crop rotation, till under plant debris, and avoid overhead watering. Use mulch to prevent soil from splashing onto lower leaves.",
            "cure": "Use fungicides containing chlorothalonil or copper when first symptoms appear and as part of a regular spray program."
        },
        "Spider Mites (Two-spotted Spider Mite)": {
            "disease": "The two-spotted spider mite, Tetranychus urticae, is a tiny spider-like pest causing yellow stippling on leaves by sucking plant juices.",
            "occurrence": "Hot, dry conditions are favorable for rapid population growth.",
            "prevention": "Use reflective mulches to deter mites. Maintaining high humidity can discourage mite outbreaks.",
            "cure": "Introduce natural predators like ladybugs or use miticides as necessary. Ensure to alternate miticides to prevent resistance development."
        },
        "Target Spot": {
            "disease": "Caused by the fungus Corynespora cassiicola, it includes symptoms of dark, target-like spots on leaves, leading to significant defoliation.",
            "occurrence": "The fungus thrives in warm, wet conditions and is spread by splashing water.",
            "prevention": "Use disease-free seeds, avoid overhead watering, and ensure good air flow around plants.",
            "cure": "Apply suitable fungicides and remove severely affected foliage. Crop rotation can help break the disease cycle."
        },
        "Yellow Leaf Curl Virus (TYLCV)": {
            "disease": "A viral disease transmitted by the whitefly, Bemisia tabaci, causing leaves to yellow, curl upwards, and stunted plant growth.",
            "occurrence": "Whiteflies carry the virus from infected plants to healthy ones. Warm climates favor whitefly populations.",
            "prevention": "Use insect-proof netting for young plants, control whiteflies with insecticides or natural predators, and remove weeds that can host whiteflies.",
            "cure": "There's no cure for infected plants. Management focuses on prevention, controlling whiteflies, and removing infected plants to prevent spread."
        },
        "Mosaic Virus": {
            "disease": "Tomato Mosaic Virus (ToMV) and Tobacco Mosaic Virus (TMV) can cause mottling, distortion of leaves, and stunted growth.",
            "occurrence": "Spread through mechanical transmission (tools, hands) and sometimes by seed. It's highly contagious.",
            "prevention": "Use virus-free seeds, sterilize tools, and avoid handling plants when wet.",
            "cure": "No cure exists. Infected plants should be removed and destroyed to prevent spread. Practice strict sanitation to control the virus."
        },
        "Healthy": {
            "disease": "Tomato Mosaic Virus (ToMV) and Tobacco Mosaic Virus (TMV) can cause mottling, distortion of leaves, and stunted growth.",
            "occurrence": "Spread through mechanical transmission (tools, hands) and sometimes by seed. It's highly contagious.",
            "prevention": "Use virus-free seeds, sterilize tools, and avoid handling plants when wet.",
            "cure": "No cure exists. Infected plants should be removed and destroyed to prevent spread. Practice strict sanitation to control the virus."
            # "description": "Plants that are not exhibiting any of the symptoms of the diseases or pest issues above, displaying vigorous growth, green leaves, and healthy fruit production."
        }
    },
    ###########CORN
    "corn" : {
        "Northern Corn Leaf Blight": {
            "disease": "Northern Corn Leaf Blight is caused by the fungus Exserohilum turcicum. It is characterized by long, slender, gray to tan lesions on the leaves, which can reduce photosynthesis and weaken the plant.",
            "occurrence": "The fungus thrives in cool, wet conditions. Spores are spread by wind and rain, infecting leaves and perpetuating the disease cycle.",
            "prevention": [
                "Plant resistant corn varieties.",
                "Practice crop rotation with non-host crops to reduce the fungal load in the soil.",
                "Keep fields clean of plant debris which may harbor the fungus."
            ],
            "cure": "Fungicide applications can be effective, particularly when applied shortly after the first signs of disease appear and before it spreads significantly."
        },
        "Common Rust": {
            "disease": "Common Rust is caused by the fungus Puccinia sorghi. It produces powdery, reddish-brown pustules on both surfaces of the corn leaves.",
            "occurrence": "The fungus is airborne and can travel long distances. It prefers cooler temperatures and moist environments.",
            "prevention": [
                "Plant rust-resistant corn varieties.",
                "Monitor fields regularly for early detection."
            ],
            "cure": "Apply fungicides as soon as rust pustules are observed, especially during cool, wet growing seasons to protect uninfected plants."
        },
        "Gray Leaf Spot": {
            "disease": "Gray Leaf Spot is caused by the fungus Cercospora zeae-maydis. It leads to rectangular, grayish lesions on the leaves, which can coalesce and destroy large portions of leaf tissue.",
            "occurrence": "The fungus overwinters in corn residue on the soil surface. Spores are spread by wind and rain splash.",
            "prevention": [
                "Practice crop rotation with non-hosts to decrease available inoculum.",
                "Bury crop residue to reduce spore survival.",
                "Plant resistant varieties when available."
            ],
            "cure": "Fungicides should be applied preventatively or at the earliest sign of disease spread. Application timing and the choice of fungicide can significantly influence the effectiveness of disease management."
        },
        "Healthy Corn Plants": {
            "disease": "Gray Leaf Spot is caused by the fungus Cercospora zeae-maydis. It leads to rectangular, grayish lesions on the leaves, which can coalesce and destroy large portions of leaf tissue.",
            "occurrence": "The fungus overwinters in corn residue on the soil surface. Spores are spread by wind and rain splash.",
            "prevention": [
                "Practice crop rotation with non-hosts to decrease available inoculum.",
                "Bury crop residue to reduce spore survival.",
                "Plant resistant varieties when available."
            ],
            "cure": "Fungicides should be applied preventatively or at the earliest sign of disease spread. Application timing and the choice of fungicide can significantly influence the effectiveness of disease management."
            # "characteristics": [
            #     "A healthy corn plant has a strong, upright growth habit.",
            #     "Leaves are vibrant green without spots or lesions.",
            #     "The plant develops fully, reaches maturity appropriately, and yields healthy ears of corn."
            # ],
            # "maintenance": [
            #     "Follow good agricultural practices, including proper spacing to ensure good air circulation.",
            #     "Maintain soil health through regular application of appropriate fertilizers and organic matter.",
            #     "Ensure adequate but not excessive irrigation to avoid water stress or overly wet conditions which can promote disease.",
            #     "Regularly monitor plant health to catch and address potential issues early."
            # ]
        }
    },
    #####################TEA
    "tea" : {
        "Anthracnose": {
            "disease": "Caused by various species of the genus Colletotrichum, this fungal disease leads to dark, sunken lesions on leaves, stems, and fruits.",
            "occurrence": "Spores spread via water, infected tools, and wind. High humidity and warm temperatures encourage development.",
            "prevention": [
                "Space plants to improve air circulation.",
                "Prune infected parts promptly.",
                "Use disease-free planting material and ensure field sanitation."
            ],
            "cure": "Apply fungicides containing copper or systemic fungicides as per guidance. Integrated disease management practices are recommended for long-term control."
        },
        "Algal Leaf Spot (Cephaleuros)": {
            "disease": "Caused by green algae Cephaleuros virescens, it manifests as raised, fuzzy, green to orange spots on the upper surface of leaves.",
            "occurrence": "The algae thrive in warm, humid, and shaded conditions, often attacking weak plants.",
            "prevention": [
                "Increase sunlight and air flow by properly spacing and pruning trees.",
                "Control plant vigor through balanced fertilization."
            ],
            "cure": "Copper-based fungicides can help control severe outbreaks. Improving plant vigor through appropriate cultural practices is also effective."
        },
        "Bird Eye Spot": {
            "disease": "Caused by the fungus Cercospora angolensis, characterized by small, round spots with a tan or grey center and a reddish-brown edge, resembling a bird's eye.",
            "occurrence": "Fungal spores spread through water splash and infect leaves under wet conditions.",
            "prevention": [
                "Improve air circulation and reduce leaf wetness to decrease infection chances.",
                "Practice good sanitation."
            ],
            "cure": "Apply fungicides as necessary, especially during wet conditions that favor the spread of this disease."
        },
        "Brown Blight": {
            "disease": "Caused by the fungus Blumeria graminis, this disease presents as brown or reddish spots or patches on the tea leaves.",
            "occurrence": "The fungus thrives and spreads rapidly in cool, humid environments.",
            "prevention": [
                "Ensure good air circulation around plants.",
                "Avoid excessive nitrogen fertilization which can increase susceptibility."
            ],
            "cure": "Apply sulfur-based or other appropriate fungicides following expert recommendations. Regular monitoring and immediate action upon first signs are crucial."
        },
        "Grey Blight": {
            "disease": "Caused by the fungus Pestalotiopsis theae, it causes small, circular, grey spots with a dark margin on leaves.",
            "occurrence": "The fungus favors warm, wet conditions, especially in crowded plantings with poor air flow.",
            "prevention": [
                "Space plants adequately to improve air circulation.",
                "Practice regular pruning and remove infected leaves to reduce fungal spread."
            ],
            "cure": "Fungicides effective against a broad range of fungi can be used following the labels and expert advice. Enhanced field hygiene and cultural control are foundational."
        },
        "Red Leaf Spot": {
            "disease": "This fungal disease, often caused by Exobasidium vexans, results in bright red spots on the leaves, causing a reduction in photosynthesis and overall vigor.",
            "occurrence": "Spread through spores via air and rain, it proliferates in humid, cooler weather.",
            "prevention": [
                "Regularly inspect plants and remove the initial infections promptly.",
                "Avoid overhead watering to reduce leaf wetness."
            ],
            "cure": "Use protective fungicides as a preventive measure in areas with a history of the disease. Improving plant resilience through proper nutrition and management is also beneficial."
        },
        "White Spot": {
            "disease": "Caused by the fungus Corticium salmonicolor, it produces white to pinkish patches on the surface of leaves.",
            "occurrence": "The fungus can spread through direct contact and by tools or equipment. High humidity and temperatures promote its growth.",
            "prevention": [
                "Keep pruning tools sterilized.",
                "Maintain lower humidity levels around plants through adequate-spacing and proper irrigation techniques."
            ],
            "cure": "Infected branches should be pruned and destroyed. Apply fungicides that are effective against a broad spectrum of fungi as a protective measure."
        },
        "Maintaining Healthy Tea Plants": {
            "disease": "Caused by the fungus Corticium salmonicolor, it produces white to pinkish patches on the surface of leaves.",
            "occurrence": "The fungus can spread through direct contact and by tools or equipment. High humidity and temperatures promote its growth.",
            "prevention": [
                "Keep pruning tools sterilized.",
                "Maintain lower humidity levels around plants through adequate-spacing and proper irrigation techniques."
            ],
            "cure": "Infected branches should be pruned and destroyed. Apply fungicides that are effective against a broad spectrum of fungi as a protective measure."
            # "characteristics": [
            #     "Strong, robust growth, with vibrant green, unblemished leaves."
            # ],
            # "maintenance": [
            #     "Implement an integrated pest management approach.",
            #     "Ensure adequate spacing, proper nutrition, and appropriate water management.",
            #     "Regular monitoring and timely action against early signs of disease or infestation contribute significantly to maintaining healthy tea gardens."
            # ]
        }
    },
    ######################################COTTON
    "cotton" : {
        "Cotton Leaf Curl Virus (CLCuV)": {
            "disease": "Cotton leaf curl virus is a devastating disease caused by a complex of virus species in the genus Begomovirus, which is transmitted by whitefly (Bemisia tabaci). It leads to a curling of leaves, stunted plant growth, and can dramatically reduce yields.",
            "occurrence": "The disease is primarily spread by the whitefly, which acquires the virus from infected plants and transmits it to healthy ones during feeding. The high population of whiteflies and the presence of infected plants nearby increase the risk of spread.",
            "prevention": [
                "Control whitefly populations through integrated pest management (IPM) strategies, including the use of reflective mulches to deter whiteflies, planting whitefly-resistant cotton varieties, and utilizing biological controls like natural predators.",
                "Remove and destroy infected plants to reduce virus sources.",
                "Avoid planting cotton back-to-back in the same field to reduce the buildup of pests and diseases."
            ],
            "cure": "There's no cure for plants already infected with CLCuV. Management focuses on preventing the spread of the disease and controlling whitefly populations."
        },
        "Cotton Bacterial Blight": {
            "disease": "Also known as angular leaf spot or cotton black arm, bacterial blight is caused by Xanthomonas citri subsp. malvacearum. The disease manifests as water-soaked lesions on leaves, black arm on stems and branches, and boll rot, severely affecting cotton yield and quality.",
            "occurrence": "The bacteria can survive in cotton seeds and plant debris, spreading through water splash, rain, and mechanical means. Warm, wet weather fosters the development and spread of the disease.",
            "prevention": [
                "Plant certified, disease-free seeds.",
                "Use resistant cotton varieties if available.",
                "Rotate crops with non-host crops to reduce the bacterial load in the soil.",
                "Practice good field sanitation by removing and destroying infected plant residues."
            ],
            "cure": "Infected fields should be treated with appropriate bactericides, following recommended guidelines. However, prevention and use of resistant varieties remain the most effective management strategies. Implementing wide row spacing to improve air circulation and reduce leaf wetness can help minimize the conditions conducive to bacterial propagation."
        },
        "Healthy Cotton Plants": {
            "disease": "Also known as angular leaf spot or cotton black arm, bacterial blight is caused by Xanthomonas citri subsp. malvacearum. The disease manifests as water-soaked lesions on leaves, black arm on stems and branches, and boll rot, severely affecting cotton yield and quality.",
            "occurrence": "The bacteria can survive in cotton seeds and plant debris, spreading through water splash, rain, and mechanical means. Warm, wet weather fosters the development and spread of the disease.",
            "prevention": [
                "Plant certified, disease-free seeds.",
                "Use resistant cotton varieties if available.",
                "Rotate crops with non-host crops to reduce the bacterial load in the soil.",
                "Practice good field sanitation by removing and destroying infected plant residues."
            ],
            "cure": "Infected fields should be treated with appropriate bactericides, following recommended guidelines. However, prevention and use of resistant varieties remain the most effective management strategies. Implementing wide row spacing to improve air circulation and reduce leaf wetness can help minimize the conditions conducive to bacterial propagation."
            # "characteristics": [
            #     "Fresh cotton leaves are typically broad, healthy green, and free of spots or deformities.",
            #     "A healthy cotton plant displays vigorous growth, has a strong stem, and produces numerous, well-formed bolls."
            # ],
            # "maintenance": [
            #     "Regular monitoring for pests and diseases and implementing IPM practices.",
            #     "Ensuring balanced fertility to promote healthy plant growth without encouraging conditions favorable to pests and diseases.",
            #     "Providing adequate irrigation, avoiding water stress, and optimizing plant density and row spacing to improve air flow and reduce disease pressure."
            # ]
        }
    },
    ###############POTATO
    "potato" : {
        "Early Blight": {
            "disease": "Early blight is a common potato disease caused by the fungus Alternaria solani. It is characterized by small, dark spots on older leaves, which can expand into larger rings, giving a target-like appearance. The disease can also affect the stems and tubers, leading to reduced yield and quality.",
            "occurrence": "The fungus overwinters in soil and plant debris, becoming active in warm, humid conditions. It spreads through rain splash, irrigation, and contaminated equipment.",
            "prevention": [
                "Rotate crops with non-hosts for at least three years to reduce soil inoculum levels.",
                "Practice good field sanitation by removing and destroying infected plant debris.",
                "Use certified, disease-free seed potatoes.",
                "Manage irrigation to avoid prolonged leaf wetness and reduce humidity in the canopy.",
                "Apply a preventive fungicide regimen, particularly during conditions that favor the disease."
            ],
            "cure": "Once symptoms are observed, apply recommended fungicides to protect uninfected foliage. Products containing chlorothalonil or mancozeb are commonly used. Be sure to rotate fungicide groups to prevent resistance development."
        },
        "Late Blight": {
            "disease": "Late blight is a potentially devastating disease of potatoes caused by the oomycete Phytophthora infestans. It can destroy leaves, stems, and tubers quickly under favorable conditions. The disease is most infamous for causing the Irish Potato Famine. Symptoms include rapid blight of foliage, dark lesions on stems, and dark, firm rot of tubers.",
            "occurrence": "The pathogen thrives in cool, wet weather conditions and spreads through wind-driven rain, equipment, and infected plant material. Spores can travel long distances through the air and infect fields far from the initial source.",
            "prevention": [
                "Implement crop rotation and avoid planting potatoes in fields adjacent to last year’s potato or tomato crops.",
                "Destroy volunteer potatoes and solanaceous weeds that can harbor the pathogen.",
                "Use resistant potato varieties whenever possible.",
                "Employ targeted irrigation practices to minimize leaf wetness and humidity.",
                "Apply a protective fungicide before the disease appears, especially when conditions are favorable for disease development."
            ],
            "cure": "Once detected, apply a specific fungicide for late blight management. Products containing metalaxyl or mefenoxam can be effective, but resistance is a concern, so these should be used in rotation with other modes of action. Infected plants should be removed and destroyed to prevent further spread."
        },
        "Healthy Potato Plants": {
            "disease": "Late blight is a potentially devastating disease of potatoes caused by the oomycete Phytophthora infestans. It can destroy leaves, stems, and tubers quickly under favorable conditions. The disease is most infamous for causing the Irish Potato Famine. Symptoms include rapid blight of foliage, dark lesions on stems, and dark, firm rot of tubers.",
            "occurrence": "The pathogen thrives in cool, wet weather conditions and spreads through wind-driven rain, equipment, and infected plant material. Spores can travel long distances through the air and infect fields far from the initial source.",
            "prevention": [
                "Implement crop rotation and avoid planting potatoes in fields adjacent to last year’s potato or tomato crops.",
                "Destroy volunteer potatoes and solanaceous weeds that can harbor the pathogen.",
                "Use resistant potato varieties whenever possible.",
                "Employ targeted irrigation practices to minimize leaf wetness and humidity.",
                "Apply a protective fungicide before the disease appears, especially when conditions are favorable for disease development."
            ],
            "cure": "Once detected, apply a specific fungicide for late blight management. Products containing metalaxyl or mefenoxam can be effective, but resistance is a concern, so these should be used in rotation with other modes of action. Infected plants should be removed and destroyed to prevent further spread.",
            # "characteristics": [
            #     "Healthy potato plants are vibrant green with sturdy stems and well-formed leaves.",
            #     "The plants produce a robust yield of tubers that are firm, unblemished, and free of rot."
            # ],
            # "maintenance": [
            #     "Select high-quality, certified disease-free seed potatoes.",
            #     "Provide adequate spacing between plants to ensure good air circulation.",
            #     "Manage irrigation to avoid excessive soil moisture and prevent standing water in fields.",
            #     "Monitor crops regularly for early signs of disease and pest activity.",
            #     "Implement a balanced fertilization program to support healthy growth without overstimulating foliage, which can be more susceptible to diseases."
            # ]
        }
    },

    ###########MANGO
    "mango" : {
        "Anthracnose": {
            "disease": "Anthracnose in mango is caused by the fungus Colletotrichum gloeosporioides. It is one of the most common and serious diseases in mango, affecting flowers, leaves, and fruit. It manifests as black, sunken lesions on the fruit, leading to significant post-harvest losses.",
            "occurrence": "The fungus thrives in moist, warm conditions and can spread rapidly. It overwinters in infected plant debris and is spread by rain splash and contaminated equipment.",
            "prevention": [
                "Prune trees to improve air circulation.",
                "Remove and destroy fallen leaves and fruits, which can harbor the fungus.",
                "Apply protective fungicides during the flowering period and pre-harvest, following local guidelines for application and timing."
            ],
            "cure": "Use registered fungicides as a post-harvest treatment for fruits. Pre-harvest, apply fungicides that are effective against anthracnose following recommended guidelines."
        },
        "Bacterial Canker": {
            "disease": "Bacterial canker in mango is caused by Xanthomonas axonopodis pv. mangiferaeindicae. It leads to cankers on stems, branches, and fruit, and can cause leaf spotting and gummosis.",
            "occurrence": "The bacteria enter the plant through wounds or natural openings and are spread by rain, insects, and contaminated tools.",
            "prevention": [
                "Avoid mechanical injuries to trees.",
                "Practice good sanitation by removing and burning infected plant parts.",
                "Use copper-containing bactericides as a preventive measure, especially after pruning."
            ],
            "cure": "There is no cure for trees once they are heavily infected; however, the progression can be slowed with the application of copper-based bactericides. Severely diseased limbs should be pruned and burned."
        },
        "Cutting Weevil": {
            "disease": "The mango cutting weevil (Sternochetus mangiferae) attacks the stems of young shoots and can bore into the fruit, causing direct damage and making the fruit vulnerable to secondary infections.",
            "occurrence": "Weevils lay their eggs in the bark of mango trees. The larvae then bore into the plant tissues, which can kill young shoots or cause fruit to drop prematurely.",
            "prevention": [
                "Regularly monitor trees, especially during the flowering and fruiting seasons.",
                "Maintain good orchard sanitation by removing and destroying infested plant parts."
            ],
            "cure": "Use insecticides recommended for weevil control in mangoes. Pesticide application timing and methods should follow expert recommendations for effective management."
        },
        "Die Back": {
            "disease": "Die back is a condition in mango trees where branches start drying back from the tips owing to fungal infections, primarily caused by Lasiodiplodia theobromae, among others.",
            "occurrence": "The fungus enters the tree through wounds and spreads during moist, humid conditions, causing the tissue to die back.",
            "prevention": [
                "Avoid injuries to the trees.",
                "Prune infected branches well below the last visible signs of the disease and burn them.",
                "Paint the wounds with fungicide to prevent infection."
            ],
            "cure": "Similarly, to prevention, infected parts of the tree should be removed. Apply fungicides to protect healthy tissues and promote recovery."
        },
        'Gall Midge': {
            "disease": "The mango gall midge (Procontarinia matteiana) causes the formation of galls on leaves and sometimes on flowers, affecting the photosynthetic capability of the plant and the overall yield.",
            "occurrence": "Midge larvae feed on the plant tissues, leading to gall formation. The pest is more prevalent in wet conditions.",
            "prevention": [
                "Prune and destroy affected plant parts to reduce the population of midges.",
                "Encourage natural predators by maintaining a biodiversity-friendly environment around the orchard."
            ],
            "cure": "Application of appropriate insecticides can help control the midge population. It's crucial to follow local guidelines for the timing and use of these products."
        },
        "Powdery Mildew": {
            "disease": "Powdery mildew, caused by the fungus Oidium mangiferae, affects leaves, flowers, and young fruits of mango trees, covering them in a white, powdery fungal growth.",
            "occurrence": "The fungus prefers high humidity and moderate temperatures. Spores are airborne and can infect other trees over long distances.",
            "prevention": [
                "Ensure good air circulation through regular pruning.",
                "Apply sulfur or other recommended fungicides at the first sign of the disease."
            ],
            "cure": "Fungicide applications should be made at the correct time and rate to effectively control the disease while minimizing impact on beneficial organisms."
        },
        "Sooty Mould": {
            "disease": "Sooty mould is a fungus that grows on honeydew secreted by insects such as aphids, scale insects, and mealybugs. It forms a black coating on leaves, significantly reducing photosynthesis.",
            "occurrence": "The condition is secondary to insect infestations that produce honeydew, on which the sooty mould fungus thrives.",
            "prevention": [
                "Control the primary insect infestation to prevent sooty mould development.",
                "Apply insecticides to manage populations of aphids, scales, and mealybugs."
            ],
            "cure": "Once the causative insect infestation is controlled, sooty mould can be washed off the leaves with a strong jet of water or treated with neem oil or similar products to help remove the fungus."
        }
    },
    ##########RICE
    "rice" : {
        "Bacterial Leaf Blight": {
            "disease": "Bacterial leaf blight is caused by the bacterium Xanthomonas oryzae pv. oryzae. It is characterized by wilting of seedlings and yellowing and drying of leaves, starting from the tips and margins, progressing inward. In severe cases, it can cause significant yield loss.",
            "occurrence": "The bacteria spread primarily through infected seeds, water splash, and contaminated tools. It enters the plant through wounds or natural openings and is favored by high humidity and temperatures between 25°C to 30°C (77°F to 86°F).",
            "prevention": [
                "Use certified disease-free seeds.",
                "Manage field water diligently to avoid excessive moisture.",
                "Practice crop rotation with non-host crops to break the disease cycle.",
                "Ensure balanced fertilization, avoiding excessive nitrogen application."
            ],
            "cure": "Once the disease is established, the focus shifts to management as no direct cure is available. Apply copper-based bactericides or antibiotics such as streptomycin sulfate following local extension recommendations, though their effectiveness can vary. Removing and destroying infected plant debris can help reduce the spread."
        },
        "Brown Spot": {
            "disease": "Brown spot is caused by the fungus Helminthosporium oryzae (also known as Cochliobolus miyabeanus). Symptoms include circular or oval brown spots on leaves, grains, and glumes, which can merge and cover large areas. This disease can lead to reduced seed quality and grain yield.",
            "occurrence": "The fungus persists in infected plant residues and soil. It spreads through air, seeds, and water splash. The disease is more severe under conditions of nutrient deficiency, particularly silicon and potassium, and in fields with poor drainage.",
            "prevention": [
                "Use disease-free seeds and treat seeds before planting.",
                "Maintain balanced soil fertility to promote healthy plant growth.",
                "Practice crop rotation and remove infected plant debris to reduce inoculum sources."
            ],
            "cure": "Fungicidal sprays can help control the disease, especially when applied at the first sign of infection. Products containing propiconazole or azoxystrobin have been effective. Enhancing plant vigor through appropriate fertilization, especially with potassium and silicon, can also reduce disease severity."
        },
        "Leaf Smut": {
            "disease": "Leaf smut of rice is caused by the fungus Entyloma oryzae. It produces long, narrow, black or dark-brown lesions on the leaves, which can reduce photosynthetic area and, thus, affect plant growth and yield.",
            "occurrence": "The fungus survives in infected crop residues and soil. It infects new plants through water splash or direct contact. The disease thrives in warm, humid environments.",
            "prevention": [
                "Plant smut-resistant rice varieties, where available.",
                "Rotate crops to reduce the buildup of fungal spores in the soil.",
                "Remove and destroy infected plant material at the end of the growing season."
            ],
            "cure": "Applying protective fungicides as a foliar spray can help manage leaf smut, though resistant cultivars and cultural practices are the primary methods of control. Ensuring plants are not stressed, especially through proper nutrition, can help reduce the impact of the disease."
        }
    }
}








if __name__ == '__main__':
    app.run(debug=True)
