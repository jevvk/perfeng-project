python3 -m pip install youtube-dl
youtube-dl https://www.youtube.com/watch?v=km2OPUctni4
mv 'Saitama vs Genos Fight _ One Punch Man (60FPS)-km2OPUctni4.mp4' video.mp4
mkdir images
ffmpeg -i video.mp4 -vf select="between(n\,0\,999),setpts=PTS-STARTPTS" images/$img%03d.bmp
