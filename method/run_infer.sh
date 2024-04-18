# python infer_sdr2hdr_video.py --file_name $input/LY-EP17-夜外红色溢色-SDR.mov --save_name $output/LY-EP17-夜外红色溢色-SDR_davinci.mp4 --model_path params.pth
input=$1
output=$2
for file in `ls $input`;
do
    python infer_sdr2hdr_video.py --file_name $input/$file --save_name $output/${file%.*}_youtu.mp4 --model_path params.pth
    python infer_sdr2hdr_video.py --file_name $input/$file --save_name $output/${file%.*}_davinci.mp4 --model_path params_DaVinci.pth
    python infer_sdr2hdr_video.py --file_name $input/$file --save_name $output/${file%.*}_dm.mp4 --model_path params_3DM.pth
done