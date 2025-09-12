from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RandomHeight, RandomWidth, RandomRotation, RandomZoom, RandomFlip
import os
from werkzeug.utils import secure_filename
import tempfile
import uuid
from PIL import Image
import io

app = Flask(__name__)

# Configure GPU memory growth to avoid allocating all GPU memory at once
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'GPU acceleration enabled: {len(gpus)} GPU(s) detected')
    except RuntimeError as e:
        print(f'GPU configuration error: {e}')
else:
    print('No GPU detected, using CPU')

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
    'rice': 'rice_densenet_finetuned_model.h5'
}

_model_cache = {}

def warmup_model(model):
    """Warm up the model with a dummy prediction"""
    try:
        dummy_input = np.random.random((1, 224, 224, 3)).astype('float32')
        _ = model.predict(dummy_input, verbose=0)
        print("Model warmup completed")
    except Exception as e:
        print(f"Model warmup failed: {e}")

def load_model(plant_type):
    # Return cached model if available
    if plant_type in _model_cache:
        return _model_cache[plant_type]

    model_filename = model_mapping.get(plant_type)
    if not model_filename:
        raise ValueError('Invalid plant type selected')

    model_path = os.path.join(MODEL_DIR, model_filename)
    print(f"Loading {plant_type} model from {model_path}")
    
    # Register custom objects for data augmentation layers
    custom_objects = {
        'RandomHeight': RandomHeight,
        'RandomWidth': RandomWidth,
        'RandomRotation': RandomRotation,
        'RandomZoom': RandomZoom,
        'RandomFlip': RandomFlip
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    # Optimize for inference
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Warm up the model
    warmup_model(model)
    
    _model_cache[plant_type] = model
    print(f"{plant_type} model loaded and optimized successfully")
    return model

def _is_allowed_file(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS

def preload_all_models():
    """Pre-load all models at startup to avoid loading delays during prediction"""
    print("Pre-loading all models...")
    for plant_type in model_mapping.keys():
        try:
            load_model(plant_type)
            print(f"✓ Loaded {plant_type} model successfully")
        except Exception as e:
            print(f"✗ Failed to load {plant_type} model: {e}")
    print("Model pre-loading completed!")

# Simple bilingual disease details
diseases_details = {
    "tomato": {
        "Bacterial Spot": {
            "disease": {
                "en": "Caused by several species of Xanthomonas, it creates small, water-soaked spots on leaves, stems, and fruits, eventually turning into scab-like lesions.",
                "bn": "বিভিন্ন প্রজাতির Xanthomonas ব্যাকটেরিয়া দ্বারা সৃষ্ট, এটি পাতায়, কাণ্ডে এবং ফলে ছোট, পানিতে ভেজা দাগ তৈরি করে, যা শেষ পর্যন্ত খোসার মতো ক্ষত হয়ে যায়।"
            },
            "occurrence": {
                "en": "Spread through water (rain or irrigation) and contaminated tools/equipment. It thrives in warm, moist conditions.",
                "bn": "পানি (বৃষ্টি বা সেচ) এবং দূষিত সরঞ্জামের মাধ্যমে ছড়ায়। এটি গরম, আর্দ্র পরিবেশে ভালো জন্মে।"
            },
            "prevention": {
                "en": "Use resistant varieties, apply copper-based sprays early, and practice crop rotation. Avoid working in fields when plants are wet to prevent spread.",
                "bn": "প্রতিরোধী জাত ব্যবহার করুন, তাড়াতাড়ি তামা-ভিত্তিক স্প্রে প্রয়োগ করুন এবং ফসল আবর্তন অনুশীলন করুন। ছড়ানো প্রতিরোধ করতে গাছ ভেজা থাকলে ক্ষেতে কাজ করা এড়িয়ে চলুন।"
            },
            "cure": {
                "en": "Once established, control is difficult. Copper sprays and improved cultural practices can mitigate but not eliminate the disease.",
                "bn": "একবার প্রতিষ্ঠিত হলে নিয়ন্ত্রণ করা কঠিন। তামা স্প্রে এবং উন্নত সাংস্কৃতিক অনুশীলন রোগ কমাতে পারে কিন্তু নির্মূল করতে পারে না।"
            }
        },
        "Early Blight": {
            "disease": {
                "en": "Caused by the fungus Alternaria solani, it manifests as dark, concentric rings on older leaves and stems, leading to leaf drop.",
                "bn": "Alternaria solani ছত্রাক দ্বারা সৃষ্ট, এটি পুরানো পাতায় এবং কাণ্ডে গাঢ়, কেন্দ্রীয় বলয়ের আকারে প্রকাশ পায়, যা পাতার ঝরে পড়ার কারণ হয়।"
            },
            "occurrence": {
                "en": "The fungus overwinters in the soil and plant debris, infecting through splashing rain or irrigation.",
                "bn": "ছত্রাক মাটি এবং গাছের ধ্বংসাবশেষে শীতকাল কাটায়, বৃষ্টির ছিটা বা সেচের মাধ্যমে সংক্রমণ ঘটায়।"
            },
            "prevention": {
                "en": "Rotate crops, till under infected debris, and practice good weed management. Use fungicide sprays when conditions favor disease development.",
                "bn": "ফসল আবর্তন করুন, সংক্রামিত ধ্বংসাবশেষ চাষ করুন এবং ভালো আগাছা ব্যবস্থাপনা অনুশীলন করুন। রোগের বিকাশের অনুকূল অবস্থায় ছত্রাকনাশক স্প্রে ব্যবহার করুন।"
            },
            "cure": {
                "en": "Apply fungicides based on mancozeb or chlorothalonil, especially after periods of rain or heavy dew.",
                "bn": "বৃষ্টি বা ভারী শিশিরের পর বিশেষ করে mancozeb বা chlorothalonil ভিত্তিক ছত্রাকনাশক প্রয়োগ করুন।"
            }
        },
        "Late Blight": {
            "disease": {
                "en": "A serious disease caused by Phytophthora infestans, leading to rapid wilting, brown lesions on leaves, stems, and fruit.",
                "bn": "Phytophthora infestans দ্বারা সৃষ্ট একটি গুরুতর রোগ, যা পাতায়, কাণ্ডে এবং ফলে দ্রুত শুকিয়ে যাওয়া, বাদামি ক্ষত সৃষ্টি করে।"
            },
            "occurrence": {
                "en": "It spreads through spores in moist, cool weather and can devastate crops quickly.",
                "bn": "এটি আর্দ্র, ঠান্ডা আবহাওয়ায় স্পোরের মাধ্যমে ছড়ায় এবং দ্রুত ফসল ধ্বংস করতে পারে।"
            },
            "prevention": {
                "en": "Grow resistant varieties, improve air circulation with proper spacing, and avoid overhead irrigation. Use fungicides preventatively in high-risk areas.",
                "bn": "প্রতিরোধী জাত চাষ করুন, উপযুক্ত দূরত্ব দিয়ে বায়ু চলাচল উন্নত করুন এবং উপর থেকে সেচ এড়িয়ে চলুন। উচ্চ ঝুঁকিপূর্ণ এলাকায় প্রতিরোধমূলকভাবে ছত্রাকনাশক ব্যবহার করুন।"
            },
            "cure": {
                "en": "Apply specific fungicides promptly at the sign of outbreak. Infected plants should be removed and destroyed.",
                "bn": "রোগের প্রাদুর্ভাবের লক্ষণ দেখা দিলে তাড়াতাড়ি নির্দিষ্ট ছত্রাকনাশক প্রয়োগ করুন। সংক্রামিত গাছ সরিয়ে ফেলুন এবং ধ্বংস করুন।"
            }
        },
        "Healthy": {
            "disease": {
                "en": "Healthy tomato plants show vigorous growth with green leaves and normal fruit development.",
                "bn": "সুস্থ টমেটো গাছ সবুজ পাতা এবং স্বাভাবিক ফল বিকাশের সাথে সক্রিয় বৃদ্ধি দেখায়।"
            },
            "occurrence": {
                "en": "Healthy plants grow under optimal conditions with proper care and nutrition.",
                "bn": "সুস্থ গাছ উপযুক্ত যত্ন এবং পুষ্টির সাথে সর্বোত্তম অবস্থায় বৃদ্ধি পায়।"
            },
            "prevention": {
                "en": "Maintain good soil health, proper watering, and regular monitoring for early disease detection.",
                "bn": "ভালো মাটি স্বাস্থ্য বজায় রাখুন, উপযুক্ত জল দেওয়া এবং প্রাথমিক রোগ সনাক্তকরণের জন্য নিয়মিত পর্যবেক্ষণ করুন।"
            },
            "cure": {
                "en": "Continue good cultural practices to maintain plant health.",
                "bn": "গাছের স্বাস্থ্য বজায় রাখতে ভালো সাংস্কৃতিক অনুশীলন চালিয়ে যান।"
            }
        }
    },
    "corn": {
        "Northern Corn Leaf Blight": {
            "disease": {
                "en": "Northern Corn Leaf Blight is caused by the fungus Exserohilum turcicum. It is characterized by long, slender, gray to tan lesions on the leaves.",
                "bn": "নর্দার্ন কর্ন লিফ ব্লাইট Exserohilum turcicum ছত্রাক দ্বারা সৃষ্ট। এটি পাতায় দীর্ঘ, সরু, ধূসর থেকে ট্যান ক্ষত দ্বারা চিহ্নিত।"
            },
            "occurrence": {
                "en": "The fungus thrives in cool, wet conditions. Spores are spread by wind and rain.",
                "bn": "ছত্রাক ঠান্ডা, ভেজা অবস্থায় ভালো জন্মে। স্পোর বাতাস এবং বৃষ্টি দ্বারা ছড়ায়।"
            },
            "prevention": {
                "en": "Plant resistant corn varieties and practice crop rotation with non-host crops.",
                "bn": "প্রতিরোধী ভুট্টা জাত চাষ করুন এবং অ-হোস্ট ফসলের সাথে ফসল আবর্তন অনুশীলন করুন।"
            },
            "cure": {
                "en": "Apply fungicides when first signs of disease appear.",
                "bn": "রোগের প্রথম লক্ষণ দেখা দিলে ছত্রাকনাশক প্রয়োগ করুন।"
            }
        },
        "Healthy Corn Plants": {
            "disease": {
                "en": "Healthy corn plants show vigorous growth with green leaves and well-developed ears.",
                "bn": "সুস্থ ভুট্টা গাছ সবুজ পাতা এবং ভালো বিকশিত কান্ডের সাথে সক্রিয় বৃদ্ধি দেখায়।"
            },
            "occurrence": {
                "en": "Healthy plants grow under optimal conditions with proper care and nutrition.",
                "bn": "সুস্থ গাছ উপযুক্ত যত্ন এবং পুষ্টির সাথে সর্বোত্তম অবস্থায় বৃদ্ধি পায়।"
            },
            "prevention": {
                "en": "Maintain good soil health, proper watering, and regular monitoring.",
                "bn": "ভালো মাটি স্বাস্থ্য বজায় রাখুন, উপযুক্ত জল দেওয়া এবং নিয়মিত পর্যবেক্ষণ করুন।"
            },
            "cure": {
                "en": "Continue good cultural practices to maintain plant health.",
                "bn": "গাছের স্বাস্থ্য বজায় রাখতে ভালো সাংস্কৃতিক অনুশীলন চালিয়ে যান।"
            }
        }
    },
    "tea": {
        "Anthracnose": {
            "disease": {
                "en": "Caused by various species of the genus Colletotrichum, this fungal disease leads to dark, sunken lesions on leaves, stems, and fruits.",
                "bn": "Colletotrichum গণের বিভিন্ন প্রজাতি দ্বারা সৃষ্ট, এই ছত্রাক রোগ পাতায়, কাণ্ডে এবং ফলে গাঢ়, ডুবে যাওয়া ক্ষত সৃষ্টি করে।"
            },
            "occurrence": {
                "en": "Spores spread via water, infected tools, and wind. High humidity and warm temperatures encourage development.",
                "bn": "স্পোর পানি, সংক্রামিত সরঞ্জাম এবং বাতাসের মাধ্যমে ছড়ায়। উচ্চ আর্দ্রতা এবং গরম তাপমাত্রা বিকাশকে উৎসাহিত করে।"
            },
            "prevention": {
                "en": "Space plants to improve air circulation, prune infected parts promptly, and use disease-free planting material.",
                "bn": "বায়ু চলাচল উন্নত করতে গাছের দূরত্ব বাড়ান, সংক্রামিত অংশ তাড়াতাড়ি ছাঁটাই করুন এবং রোগমুক্ত রোপণ সামগ্রী ব্যবহার করুন।"
            },
            "cure": {
                "en": "Apply fungicides containing copper or systemic fungicides as per guidance.",
                "bn": "নির্দেশনা অনুযায়ী তামা বা সিস্টেমিক ছত্রাকনাশক সমৃদ্ধ ছত্রাকনাশক প্রয়োগ করুন।"
            }
        },
        "Maintaining Healthy Tea Plants": {
            "disease": {
                "en": "Healthy tea plants show vigorous growth with green leaves and good yield.",
                "bn": "সুস্থ চা গাছ সবুজ পাতা এবং ভালো ফলনের সাথে সক্রিয় বৃদ্ধি দেখায়।"
            },
            "occurrence": {
                "en": "Healthy plants grow under optimal conditions with proper care and nutrition.",
                "bn": "সুস্থ গাছ উপযুক্ত যত্ন এবং পুষ্টির সাথে সর্বোত্তম অবস্থায় বৃদ্ধি পায়।"
            },
            "prevention": {
                "en": "Maintain good soil health, proper pruning, and regular monitoring.",
                "bn": "ভালো মাটি স্বাস্থ্য বজায় রাখুন, উপযুক্ত ছাঁটাই এবং নিয়মিত পর্যবেক্ষণ করুন।"
            },
            "cure": {
                "en": "Continue good cultural practices to maintain plant health.",
                "bn": "গাছের স্বাস্থ্য বজায় রাখতে ভালো সাংস্কৃতিক অনুশীলন চালিয়ে যান।"
            }
        }
    },
    "cotton": {
        "Cotton Leaf Curl Virus (CLCuV)": {
            "disease": {
                "en": "Cotton leaf curl virus is a devastating disease caused by a complex of virus species in the genus Begomovirus, transmitted by whitefly.",
                "bn": "কটন লিফ কার্ল ভাইরাস Begomovirus গণের ভাইরাস প্রজাতির একটি জটিল দ্বারা সৃষ্ট একটি ধ্বংসাত্মক রোগ, যা সাদা মাছি দ্বারা সং transmitted।"
            },
            "occurrence": {
                "en": "The disease is primarily spread by the whitefly, which acquires the virus from infected plants and transmits it to healthy ones.",
                "bn": "রোগটি প্রাথমিকভাবে সাদা মাছি দ্বারা ছড়ায়, যা সংক্রামিত গাছ থেকে ভাইরাস অর্জন করে এবং সুস্থ গাছে সং transmitted করে।"
            },
            "prevention": {
                "en": "Control whitefly populations through integrated pest management strategies and use resistant cotton varieties.",
                "bn": "সমন্বিত কীটপতঙ্গ ব্যবস্থাপনা কৌশলের মাধ্যমে সাদা মাছির জনসংখ্যা নিয়ন্ত্রণ করুন এবং প্রতিরোধী তুলা জাত ব্যবহার করুন।"
            },
            "cure": {
                "en": "There's no cure for plants already infected with CLCuV. Management focuses on preventing the spread of the disease.",
                "bn": "CLCuV দিয়ে ইতিমধ্যে সংক্রামিত গাছের জন্য কোনো নিরাময় নেই। ব্যবস্থাপনা রোগের ছড়ানো প্রতিরোধের উপর দৃষ্টি নিবদ্ধ করে।"
            }
        },
        "Healthy Cotton Plants": {
            "disease": {
                "en": "Healthy cotton plants show vigorous growth with green leaves and good boll development.",
                "bn": "সুস্থ তুলা গাছ সবুজ পাতা এবং ভালো বোল বিকাশের সাথে সক্রিয় বৃদ্ধি দেখায়।"
            },
            "occurrence": {
                "en": "Healthy plants grow under optimal conditions with proper care and nutrition.",
                "bn": "সুস্থ গাছ উপযুক্ত যত্ন এবং পুষ্টির সাথে সর্বোত্তম অবস্থায় বৃদ্ধি পায়।"
            },
            "prevention": {
                "en": "Maintain good soil health, proper watering, and regular monitoring.",
                "bn": "ভালো মাটি স্বাস্থ্য বজায় রাখুন, উপযুক্ত জল দেওয়া এবং নিয়মিত পর্যবেক্ষণ করুন।"
            },
            "cure": {
                "en": "Continue good cultural practices to maintain plant health.",
                "bn": "গাছের স্বাস্থ্য বজায় রাখতে ভালো সাংস্কৃতিক অনুশীলন চালিয়ে যান।"
            }
        }
    },
    "potato": {
        "Early Blight": {
            "disease": {
                "en": "Early blight is a common potato disease caused by the fungus Alternaria solani. It is characterized by small, dark spots on older leaves.",
                "bn": "আর্লি ব্লাইট Alternaria solani ছত্রাক দ্বারা সৃষ্ট আলুর একটি সাধারণ রোগ। এটি পুরানো পাতায় ছোট, গাঢ় দাগ দ্বারা চিহ্নিত।"
            },
            "occurrence": {
                "en": "The fungus overwinters in soil and plant debris, becoming active in warm, humid conditions.",
                "bn": "ছত্রাক মাটি এবং গাছের ধ্বংসাবশেষে শীতকাল কাটায়, গরম, আর্দ্র অবস্থায় সক্রিয় হয়ে ওঠে।"
            },
            "prevention": {
                "en": "Rotate crops with non-hosts for at least three years and practice good field sanitation.",
                "bn": "অন্তত তিন বছর অ-হোস্ট ফসলের সাথে আবর্তন করুন এবং ভালো ক্ষেতের স্বাস্থ্যবিধি অনুশীলন করুন।"
            },
            "cure": {
                "en": "Apply recommended fungicides to protect uninfected foliage.",
                "bn": "অ-সংক্রামিত পাতাকে রক্ষা করতে সুপারিশকৃত ছত্রাকনাশক প্রয়োগ করুন।"
            }
        },
        "Healthy Potato Plants": {
            "disease": {
                "en": "Healthy potato plants show vigorous growth with green leaves and good tuber development.",
                "bn": "সুস্থ আলু গাছ সবুজ পাতা এবং ভালো আলু বিকাশের সাথে সক্রিয় বৃদ্ধি দেখায়।"
            },
            "occurrence": {
                "en": "Healthy plants grow under optimal conditions with proper care and nutrition.",
                "bn": "সুস্থ গাছ উপযুক্ত যত্ন এবং পুষ্টির সাথে সর্বোত্তম অবস্থায় বৃদ্ধি পায়।"
            },
            "prevention": {
                "en": "Maintain good soil health, proper watering, and regular monitoring.",
                "bn": "ভালো মাটি স্বাস্থ্য বজায় রাখুন, উপযুক্ত জল দেওয়া এবং নিয়মিত পর্যবেক্ষণ করুন।"
            },
            "cure": {
                "en": "Continue good cultural practices to maintain plant health.",
                "bn": "গাছের স্বাস্থ্য বজায় রাখতে ভালো সাংস্কৃতিক অনুশীলন চালিয়ে যান।"
            }
        }
    },
    "rice": {
        "Bacterial Leaf Blight": {
            "disease": {
                "en": "Bacterial leaf blight is caused by the bacterium Xanthomonas oryzae pv. oryzae. It is characterized by wilting of seedlings and yellowing of leaves.",
                "bn": "ব্যাকটেরিয়াল লিফ ব্লাইট Xanthomonas oryzae pv. oryzae ব্যাকটেরিয়া দ্বারা সৃষ্ট। এটি চারার শুকিয়ে যাওয়া এবং পাতার হলুদ হয়ে যাওয়া দ্বারা চিহ্নিত।"
            },
            "occurrence": {
                "en": "The bacteria spread primarily through infected seeds, water splash, and contaminated tools.",
                "bn": "ব্যাকটেরিয়া প্রাথমিকভাবে সংক্রামিত বীজ, পানির ছিটা এবং দূষিত সরঞ্জামের মাধ্যমে ছড়ায়।"
            },
            "prevention": {
                "en": "Use certified disease-free seeds and manage field water diligently to avoid excessive moisture.",
                "bn": "প্রমাণিত রোগমুক্ত বীজ ব্যবহার করুন এবং অতিরিক্ত আর্দ্রতা এড়াতে ক্ষেতের পানি সতর্কতার সাথে ব্যবস্থাপনা করুন।"
            },
            "cure": {
                "en": "Apply copper-based bactericides following local extension recommendations.",
                "bn": "স্থানীয় সম্প্রসারণ সুপারিশ অনুসরণ করে তামা-ভিত্তিক ব্যাকটেরিয়ানাশক প্রয়োগ করুন।"
            }
        }
    }
}

# Disease name translations
disease_translations = {
    "Bacterial Spot": {"en": "Bacterial Spot", "bn": "ব্যাকটেরিয়াল স্পট (Bacterial Spot)"},
    "Early Blight": {"en": "Early Blight", "bn": "আর্লি ব্লাইট (Early Blight)"},
    "Late Blight": {"en": "Late Blight", "bn": "লেট ব্লাইট (Late Blight)"},
    "Healthy": {"en": "Healthy", "bn": "সুস্থ (Healthy)"},
    "Northern Corn Leaf Blight": {"en": "Northern Corn Leaf Blight", "bn": "নর্দার্ন কর্ন লিফ ব্লাইট (Northern Corn Leaf Blight)"},
    "Healthy Corn Plants": {"en": "Healthy Corn Plants", "bn": "সুস্থ ভুট্টা গাছ (Healthy Corn Plants)"},
    "Anthracnose": {"en": "Anthracnose", "bn": "অ্যানথ্রাকনোজ (Anthracnose)"},
    "Maintaining Healthy Tea Plants": {"en": "Maintaining Healthy Tea Plants", "bn": "সুস্থ চা গাছ বজায় রাখা (Maintaining Healthy Tea Plants)"},
    "Cotton Leaf Curl Virus (CLCuV)": {"en": "Cotton Leaf Curl Virus (CLCuV)", "bn": "কটন লিফ কার্ল ভাইরাস (Cotton Leaf Curl Virus)"},
    "Healthy Cotton Plants": {"en": "Healthy Cotton Plants", "bn": "সুস্থ তুলা গাছ (Healthy Cotton Plants)"},
    "Healthy Potato Plants": {"en": "Healthy Potato Plants", "bn": "সুস্থ আলু গাছ (Healthy Potato Plants)"},
    "Bacterial Leaf Blight": {"en": "Bacterial Leaf Blight", "bn": "ব্যাকটেরিয়াল লিফ ব্লাইট (Bacterial Leaf Blight)"}
}

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

            # Process image in memory instead of saving to disk
            file_data = file.read()
            
            # Load and process image in memory
            img = Image.open(io.BytesIO(file_data))
            img = img.resize((224, 224))
            img = img.convert('RGB')  # Ensure RGB format
            
            # Convert to numpy array and normalize
            x = np.array(img)
            x = np.expand_dims(x, axis=0)
            x = x.astype('float32') / 255.0

            model = load_model(plant_type)

            # Make prediction
            preds = model.predict(x, verbose=0)
            pred_class = np.argmax(preds, axis=1)

            result = get_class(plant_type, pred_class[0])

            # Get language preference from form or default to English
            language = request.form.get('language', 'en')
            
            # Get disease details in the selected language
            disease_info = diseases_details.get(plant_type, {}).get(result, {})
            translated_disease_info = {}
            for key, value in disease_info.items():
                if isinstance(value, dict) and language in value:
                    translated_disease_info[key] = value[language]
                else:
                    translated_disease_info[key] = value
            
            # Get translated disease name
            translated_result = disease_translations.get(result, {}).get(language, result)
            
            return render_template(
                'index.html',
                prediction=translated_result,
                diseases_details=translated_disease_info,
                current_language=language
            )
        except Exception as exc:
            return render_template('index.html', prediction=f'Error: {str(exc)}')
        finally:
            # No file cleanup needed since we process in memory
            pass

    # GET request
    language = request.args.get('lang', 'en')
    return render_template('index.html', prediction=None, current_language=language)

def get_class(plant_type, index):
    # Return labels that exactly match the keys used in `diseases_details` so lookups succeed
    classes = {
        'tomato': ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Healthy'],
        'corn': ['Northern Corn Leaf Blight', 'Healthy Corn Plants'],
        'tea': ['Anthracnose', 'Maintaining Healthy Tea Plants'],
        'cotton': ['Cotton Leaf Curl Virus (CLCuV)', 'Healthy Cotton Plants'],
        'potato': ['Early Blight', 'Healthy Potato Plants'],
        'rice': ['Bacterial Leaf Blight']
    }
    return classes[plant_type][index]

if __name__ == '__main__':
    print('Starting Plant Disease Prediction App...')
    print('Pre-loading models for faster predictions...')
    preload_all_models()
    print('All models loaded successfully!')
    print('Starting Flask server...')
    app.run(debug=True)