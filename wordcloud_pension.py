import re
import lemmy
import lemmy.pipe
import nltk
from polyglot.text import Text
import pycld2 as cld2
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import stopwordsiso as stopwords


lemmatizer = lemmy.load("da")

# Stop words + cumstoms
stopwordlist = stopwords.stopwords("da")
stopwordlist.update(['du','og','til','kan','vores','brug','dine','første','ved','find','dit','mere','blevet','tager','søg','http','dk','søg','læs'])

# Open file and lower case letters
with open("pfa.txt","r") as file:
    text = file.read().lower()

# Remove numbers from text
text = ''.join([i for i in text if not i.isdigit()])

# Remove all special characters
text = re.sub(r'[-()\"#_/@;:<>{}`+=~|.!?,]', ' ', text)

# Tokenize text and remove stop words
text = ' '.join([word for word in text.split() if word not in stopwordlist])

# Set text to Danish
text = Text(text, hint_language_code='da')

# Load text in to dataframe and POS-tag word
df = pd.DataFrame(text.pos_tags)

# Label each column
df.rename(columns={0:'word', 1:'Pos',}, inplace=True)

# Lemmatize function
def lemmis(row):
    return lemmatizer.lemmatize(row['Pos'], row['word'])

df['root_word'] = df.apply(lemmis, axis=1)


# Process logo
logo_mask = np.array(Image.open("velliv.png"))
image_colors = ImageColorGenerator(logo_mask)

#Convert dataframe to string
text12 = df['root_word'].to_string(index = False)

# Create and generate a word cloud image:
wc = WordCloud(background_color="white", mask=logo_mask,  max_words=1000, max_font_size=90, width=600, height=600, contour_width=0.1, contour_color='gray',
 random_state=42).generate(str(text12))

# Display the generated image:
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.show()
