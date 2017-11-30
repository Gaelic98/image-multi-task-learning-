
python3 multi_task_mnf.py --epoch=25 --load_model=False --learning_rate=0.001
python3 multi_task_mnf.py --epoch=25 --load_model=True --learning_rate=0.00001
#python3 multi_task_mn.py --epoch=25 --load_model=False --learning_rate=0.001
#python3 multi_task_mn.py --epoch=25 --load_model=True --learning_rate=0.00001
#python3 multi_task_fn.py --epoch=25 --load_model=False --learning_rate=0.001
#python3 multi_task_fn.py --epoch=25 --load_model=True --learning_rate=0.00001
#python3 multi_task_mf.py --epoch=25 --load_model=False --learning_rate=0.001
#python3 multi_task_mf.py --epoch=25 --load_model=True --learning_rate=0.00001

for i in 0 1 2
do
	python3 single_task_cnn.py --epoch=25 --load_model=False --learning_rate=0.001 --multi_task=True --datasets=$i
	python3 single_task_cnn.py --epoch=25 --load_model=True --learning_rate=0.00001 --multi_task=True --datasets=$i
done

#for ((i=0;i<=2; i=i+1))
#do
	#python3 single_task_cnn.py --epoch=25 --load_model=False --learning_rate=0.001 --multi_task=False --datasets=$i
	#python3 single_task_cnn.py --epoch=25 --load_model=True --learning_rate=0.00001 --multi_task=False --datasets=$i
#done

