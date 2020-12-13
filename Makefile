service: service.c 
	gcc service.c -o service -g -I/home/lyz/cuda_related/cuda-10.1/targets/x86_64-linux/include -L/usr/local/cuda/lib64 -libverbs -lcudart
	gcc client.c -o client -g -I/home/lyz/cuda_related/cuda-10.1/targets/x86_64-linux/include -L/usr/local/cuda/lib64 -libverbs -lcudart
clean:
	rm -rf ./*.o ./service
