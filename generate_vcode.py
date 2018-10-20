from PIL import Image, ImageFont, ImageDraw, ImageFilter
import random
import string

fontpath = '/Library/Fonts/Arial.ttf'
def rndNumber():
    return random.choice(string.digits)

def rndColor():
    return (random.randint(0,255),random.randint(0,255), random.randint(0,255))

def check_code(width=112, height=32, nchar=6, fontpath='/Library/Fonts/Arial Black.ttf'):
    code = []
    img = Image.new(mode='RGBA',size=(width,height),color=(255,255,255))
    draw = ImageDraw.Draw(img)
    for i in range(nchar):
        char = rndNumber()
        code.append(char)
        h = random.randint(2,6)
        font = ImageFont.truetype(fontpath,random.randint(25,30))
        draw.text([i*18+6,h],char, font=font,fill=rndColor())
    for i in range(5):
        x1 = random.randint(0,width/3)
        y1 = random.randint(0, height)
        x2 = random.randint(0,width/3)+width/2
        y2= random.randint(height/3,height)
        draw.line((x1,y1,x2,y2),fill=rndColor())
    img= img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    img.save('./genedata/'+''.join(code)+'.png')

def gene_text():
    code = ''.join(random.sample(string.digits,6))
    return code

def gene_code():
    width, height= 112, 32
    image = Image.new("RGBA",(width, height),(255,255,255))
    
    font = ImageFont.truetype('/Library/Fonts/Arial.ttf',25)
    draw = ImageDraw.Draw(image)
    text = gene_text()
    font_width,font_height = font.getsize(text)
    draw.text(((width-font_width)/6,(height-font_height)/6),text, font=font, fill=(255,0,0))
    gene_line(draw,width, height)
    image= image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    image.save('wtf.png')

def gene_line(draw, width, height):
    begin = (random.randint(0,width), random.randint(0,height))
    end = (random.randint(0, width), random.randint(0,height))
    draw.line([begin,end],fill=(random.randint(0,255), random.randint(0,255),random.randint(0,255)))
for i in range(10000):
    check_code()
    if i % 1000==0:
        print('generated %i images'%i)
