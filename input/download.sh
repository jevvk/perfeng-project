python3 -m pip install youtube-dl
youtube-dl -o video.mp4 https://www.youtube.com/watch?v=Q6iK6DjV_iE
mkdir images
ffmpeg -i video.mp4 -vf select="between(n\,0\,1000),setpts=PTS-STARTPTS" images/$img%03d.bmp
