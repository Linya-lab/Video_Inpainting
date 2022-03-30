folder="/data/dataset/video_inpainting/train_3"
for filename in ${folder}/*
do
    videoname=`basename ${filename%.*}`
    mkdir /data/dataset/video_inpainting/train/"${videoname}"
    /data/ppchu/ffmpeg/ffmpeg-git-20201128-amd64-static/ffmpeg -i $filename /data/dataset/video_inpainting/train/"${videoname}"/%06d.png
done