#include <cuda_runtime.h>
#include <string.h>
#include "unix_socket.h"


#define MSG "READ operation content"
#define MSG_SIZE (strlen(MSG) + 1) 
 
int main(void)
{
  //========================================================connecting unix domain sock	
  srand((int)time(0));
  int connfd; 
  connfd = unix_socket_conn("foo.sock");
  if(connfd<0)
  {
     printf("Error[%d] when connecting unix sock...",errno);
	 return 0;
  }
  printf("Unix sock connected.\n");
  //========================================================obtain CUDA mem from router
  printf("Begin to recv CUDA handle ...\n");  
  int i,n,size;
  cudaIpcMemHandle_t* container_mem_handle = (cudaIpcMemHandle_t*)malloc(sizeof(cudaIpcMemHandle_t));
  size = recv(connfd, container_mem_handle, sizeof(cudaIpcMemHandle_t), 0);
  if(size>=0)
  {
    // rvbuf[size]='\0';
     printf("Recieved CUDA handle[%d]\n",size);
  }
  if(size==-1)
  {
      printf("Error[%d] when recieving Data:%s.\n",errno,strerror(errno));
  }

   void *dataBuf;
   int mret = cudaIpcOpenMemHandle(&dataBuf,*container_mem_handle,cudaIpcMemLazyEnablePeerAccess);
   if( mret != cudaSuccess ) {
        printf("Cuda get handle failure '%s'", cudaGetErrorString(mret));
   }
   printf("Get CUDA mem success.\n");
   //========================================================copying msg to GPU
   char* msg = (char*)malloc(MSG_SIZE);
   strcpy(msg,MSG);   
   mret = cudaMemcpy(dataBuf, msg, MSG_SIZE,cudaMemcpyHostToDevice);
   if( mret != cudaSuccess ) {
        printf("Cuda mem copy failure '%s'", cudaGetErrorString(mret));
   }
   printf("Msg write to GPU.\n");   
   //========================================================signaling router to start sending data
   sleep(1);
   int signal = MSG_SIZE;
   size = send(connfd, &signal, sizeof(int), 0);
   if(size>=0)
   {
          printf("Signal[%d] Sended.\n",size);
   }
   if(size==-1)
   {
      printf("Error[%d] when Sending signal:%s.\n",errno,strerror(errno));
   }

   printf("Waiting for peer data operation, exit in 300 seconds...\n");
   sleep(300);
   unix_socket_close(connfd);
   printf("Client exited.\n");    
 }
