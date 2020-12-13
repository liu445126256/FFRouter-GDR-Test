
#include <cuda_runtime.h>
#include "unix_socket.h"
#include "ib_comm.h"

/******************************************************************************
* Function: print_config
*
* Input
* none
*
* Output
* none
*
* Returns
* none
*
* Description
* Print out config information
******************************************************************************/
static void print_config(void)
{
	fprintf(stdout, " ------------------------------------------------\n");
	fprintf(stdout, " Device name : \"%s\"\n", config.dev_name);
	fprintf(stdout, " IB port : %u\n", config.ib_port);
	if (config.server_name)
		fprintf(stdout, " IP : %s\n", config.server_name);
	fprintf(stdout, " TCP port : %u\n", config.tcp_port);
	if (config.gid_idx >= 0)
		fprintf(stdout, " GID index : %u\n", config.gid_idx);
	fprintf(stdout, " ------------------------------------------------\n\n");
}

/******************************************************************************
* Function: usage
*
* Input
* argv0 command line arguments
*
* Output
* none
*
* Returns
* none
*
* Description
* print a description of command line syntax
******************************************************************************/
static void usage(const char *argv0)
{
	fprintf(stdout, "Usage:\n");
	fprintf(stdout, " %s start a server and wait for connection\n", argv0);
	fprintf(stdout, " %s <host> connect to server at <host>\n", argv0);
	fprintf(stdout, "\n");
	fprintf(stdout, "Options:\n");
	fprintf(stdout, " -p, --port <port> listen on/connect to port <port> (default 18515)\n");
	fprintf(stdout, " -d, --ib-dev <dev> use IB device <dev> (default first device found)\n");
	fprintf(stdout, " -i, --ib-port <port> use port <port> of IB device (default 1)\n");
	fprintf(stdout, " -g, --gid_idx <git index> gid index to be used in GRH (default not used)\n");
}
/******************************************************************************
* Function: GDRreg
*
* Input
* res pointer to resources structure
* memPtr pointer to the memory region allocated using cudaMalloc() 
* size of the memory region
*
* Output
* none
*
* Returns
* 0 on success, 1 on failure
*
* Description
* Registering GPU memory to RDMA nic
******************************************************************************/
#define GPU_BUF_SIZE (2*1024*1024)
void GDRreg(struct resources *res, void* memPtr, int size)
{
	int mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    res->mr = ibv_reg_mr(res->pd, memPtr, size, mr_flags);
    if (!res->mr)
    {
        fprintf(stderr, "ibv_reg_mr failed with mr_flags=0x%x\n", mr_flags);
    }
    else{
        printf("GDR register success %u\n",(unsigned int)memPtr);
    	res->buf = memPtr;
	}
}



/******************************************************************************
* Function: main
*
* Input
* argc number of items in argv
* argv command line parameters
*
* Output
* none
*
* Returns
* 0 on success, 1 on failure
*
* Description
* Freeflow with GDR simple test
******************************************************************************/
int main(int argc, char *argv[])
{
    
	/* =====================================================================RDMA connection preparation */
	struct resources res;
	int rc = 1;
	char temp_char;
	/* parse the command line parameters */
	while (1)
	{
		int c;
		static struct option long_options[] = {
			{.name = "port", .has_arg = 1, .val = 'p'},
			{.name = "ib-dev", .has_arg = 1, .val = 'd'},
			{.name = "ib-port", .has_arg = 1, .val = 'i'},
			{.name = "gid-idx", .has_arg = 1, .val = 'g'},
			{.name = NULL, .has_arg = 0, .val = '\0'}
        };
		c = getopt_long(argc, argv, "p:d:i:g:", long_options, NULL);
		if (c == -1)
			break;
		switch (c)
		{
		case 'p':
			config.tcp_port = strtoul(optarg, NULL, 0);
			break;
		case 'd':
			config.dev_name = strdup(optarg);
			break;
		case 'i':
			config.ib_port = strtoul(optarg, NULL, 0);
			if (config.ib_port < 0)
			{
				usage(argv[0]);
				return 1;
			}
			break;
		case 'g':
			config.gid_idx = strtoul(optarg, NULL, 0);
			if (config.gid_idx < 0)
			{
				usage(argv[0]);
				return 1;
			}
			break;
		default:
			usage(argv[0]);
			return 1;
		}
	}
	/* parse the last parameter (if exists) as the server name */
	if (optind == argc - 1)
		config.server_name = argv[optind];
        if(config.server_name){
            printf("servername=%s\n",config.server_name);
        }
	else if (optind < argc)
	{
		usage(argv[0]);
		return 1;
	}

	/*alloc GPU memory*/
	void* dataBuf;
    int mret = cudaMalloc(&dataBuf, GPU_BUF_SIZE);
    if( mret != cudaSuccess ) {
        printf("Cuda malloc failure '%s'", cudaGetErrorString(mret));
    }

    /* no server name specified means this is not the peer procedure*/
    if(!config.server_name){
    	/*===================================================================== create unix domain socket */
	    int listenfd,connfd;
	    listenfd = unix_socket_listen("foo.sock");
	    if(listenfd<0)
	    {
	       printf("Error[%d] when listening unix socket...\n",errno);
	           return 0;
	    }
	    printf("Finished listening unix socket...\n");
	    uid_t uid;
	    connfd = unix_socket_accept(listenfd, &uid);
	    unix_socket_close(listenfd);
	    if(connfd<0)
	    {
	       printf("Error[%d] when accepting unix socket...\n",errno);
	       return 0;
	    }
	    printf("Unix sock connected\n");	
	  

		/* ===================================================================== sending shared cuda memory */
		cudaIpcMemHandle_t *router_mem_handle = (cudaIpcMemHandle_t *)malloc(sizeof(cudaIpcMemHandle_t));
		mret = cudaIpcGetMemHandle(router_mem_handle,dataBuf);
		if( mret != cudaSuccess ) {
	        printf("Cuda get handle failure '%s'", cudaGetErrorString(mret));
	    }
		/* send the handle to client container process */
		int size = send(connfd, router_mem_handle, sizeof(cudaIpcMemHandle_t), 0);
	    if(size>=0)
	    {
	        printf("CUDA handle[%d] Sended.\n",size);
	    }
	    if(size==-1)
	    {
	    	printf("Error[%d] when Sending CUDA handle:%s.\n",errno,strerror(errno));
	    }

		/* waiting for the signal (from client container) to send data */
		int signal = -1;
		size = recv(connfd, &signal, sizeof(int), 0);
		if(size>=0)
	    {
	    	printf("Signal[%d] Recieved, Getting ready to setup RDMA .\n",size);
	    }
	    if(size==-1)
	    {
	    	printf("Error[%d] when Receiving signal:%s.\n",errno,strerror(errno));
	    }
    }
    /* ======================================================== setup connection with peer */
	/* print the used parameters for info*/
	print_config();
	/* init all of the resources, so cleanup will be easy */
	resources_init(&res);

	/* create resources before using them */
	if (resources_create(&res))
	{
		fprintf(stderr, "failed to create resources\n");
		goto main_exit;
	}
	printf("Resources created\n");
	/* Register GPU mem to nic */
    GDRreg(&res,dataBuf,MSG_SIZE);	
	/* connect the QPs */
	if (connect_qp(&res))
	{
		fprintf(stderr, "failed to connect QPs\n");
		goto main_exit;
	}

	/* ====================================================== read/write operation from remote peer */
	/* Sync so we are sure server side has data ready before client tries to read it */
	if (sock_sync_data(res.sock, 1, "R", &temp_char)) 
	{
		fprintf(stderr, "sync error before RDMA ops\n");
		rc = 1;
		goto main_exit;
	}
	
	char *recv_msg = (char *)malloc(1024);

	/* this is the peer procedure */
	if (config.server_name)
	{
		/* First we read contens of server's buffer */
		if (post_send(&res, IBV_WR_RDMA_READ))
		{
			fprintf(stderr, "failed to post SR 2\n");
			rc = 1;
			goto main_exit;
		}
		if (poll_completion(&res))
		{
			fprintf(stderr, "poll completion failed 2\n");
			rc = 1;
			goto main_exit;
		}
		mret = cudaMemcpy(recv_msg, res.buf, MSG_SIZE,cudaMemcpyDeviceToHost);
        if( mret != cudaSuccess ) {
            printf("GPU Read failure '%s'", cudaGetErrorString(mret));
        }
        fprintf(stdout, "Contents of server's buffer: '%s'\n", recv_msg);


		/* Now we replace what's in the server's buffer */
		strcpy(recv_msg, RDMAMSGW);
		fprintf(stdout, "Now replacing it with: '%s'\n", recv_msg);
		mret = cudaMemcpy(res.buf, recv_msg, MSG_SIZE, cudaMemcpyHostToDevice);
        if( mret != cudaSuccess ) {
            printf("GPU Write failure '%s'", cudaGetErrorString(mret));
        }

		if (post_send(&res, IBV_WR_RDMA_WRITE))
		{
			fprintf(stderr, "failed to post SR 3\n");
			rc = 1;
			goto main_exit;
		}
		if (poll_completion(&res))
		{
			fprintf(stderr, "poll completion failed 3\n");
			rc = 1;
			goto main_exit;
		}
	}
	/* Sync so server will know that client is done mucking with its memory */
	if (sock_sync_data(res.sock, 1, "W", &temp_char)) /* just send a dummy char back and forth */
	{
		fprintf(stderr, "sync error after RDMA ops\n");
		rc = 1;
		goto main_exit;
	}

	if (!config.server_name){
		//read data from GPU
        mret = cudaMemcpy(recv_msg, dataBuf, MSG_SIZE,cudaMemcpyDeviceToHost);
		if( mret != cudaSuccess ) {
        	printf("GPU Read failure '%s'", cudaGetErrorString(mret));
        }
		fprintf(stdout, "Contents of server buffer: '%s'\n", recv_msg);
	}
	rc = 0;


	printf("Task finished, sleeping 60 seconds before existing...\n");
    sleep(60);

main_exit:
	/*free cuda mem*/
	mret = cudaFree(res.buf);
	if( mret != cudaSuccess ) {
    	printf("GPU Read failure '%s'", cudaGetErrorString(mret));
    }
	if (resources_destroy(&res))
	{
		fprintf(stderr, "failed to destroy resources\n");
		rc = 1;
	}
	if (config.dev_name)
		free((char *)config.dev_name);
	fprintf(stdout, "\ntest result is %d\n", rc);
	return rc;
}
