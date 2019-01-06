from starlette.applications import Starlette
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastai.vision import (
    ImageDataBunch,
    create_cnn,
    open_image,
    get_transforms,
    models,
)

import torch
from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp
import asyncio

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

app = Starlette()

path_img = Path('/tmp')
classes = ['leonberger', 'pomeranian','Bombay','yorkshire_terrier',
           'basset_hound','Sphynx','newfoundland','english_cocker_spaniel',
           'beagle','english_setter','american_pit_bull_terrier',
           'wheaten_terrier','staffordshire_bull_terrier','scottish_terrier',
           'Russian_Blue','Egyptian_Mau','Maine_Coon','Bengal','Siamese',
           'chihuahua','great_pyrenees','Ragdoll','boxer','german_shorthaired',
           'japanese_chin','american_bulldog','British_Shorthair','Persian',
           'shiba_inu','Birman','keeshond','miniature_pinscher','saint_bernard',
           'Abyssinian','samoyed','havanese','pug']

data = ImageDataBunch.single_form_classes(path_img, classes, tfms=get_transforms(),
                                          size=224)

learn = create_cnn(data, models.resnet34).load('pet-prophet-stage2.pth')

@app.route('/upload', methods=['POST'])
async def upload(request):
    data = await request.form()
    bytes = await (data['file'].read())
    return predict_image_from_bytes(bytes)

@app.route('/classify-url', methods['GET'])
async def classify_url(request):
    bytes = await get_bytes(request.query_params['url'])
    return predict_image_from_bytes(bytes)

def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    _,_,losses = learn.predict(img)
    return JSONResponse(
        {
            "predictions": sorted(
                zip(learner.data.classes, map(float, losses)),
                key = lambda p:p[1],
                reverse = True
            )
        }
    )

@app.route('/')
def form(request):
    return HTMLResponse(
        '''
            <form action="/upload" method="post" enctype="multipart/form-data">
                Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
            </form>
            Or submit a URL:
            <form action="/classify-url" method="get">
                <input type="url" name="url">
                <input type="submit" value="Fetch and analyze image">
            </form>
        '''
    )

@app.route('/form')
def redirect_to_homepage(request):
    return RedirectResponse('/')

if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app, host='0.0.0.0', port=8008)