#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>


// the max connection number of the server
#define MAX_CONNECTION_NUMBER 5


/* * Create a server endpoint of a connection. * Returns fd if all OK, <0 on error. */
int unix_socket_listen(const char *servername)
{
  int fd;
  struct sockaddr_un un;
  if ((fd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0)
  {
         return(-1);
  }
  int len, rval;
  unlink(servername);               /* in case it already exists */
  memset(&un, 0, sizeof(un));
  un.sun_family = AF_UNIX;
  strcpy(un.sun_path, servername);
  len = offsetof(struct sockaddr_un, sun_path) + strlen(servername);
  /* bind the name to the descriptor */
  if (bind(fd, (struct sockaddr *)&un, len) < 0)
  {
    rval = -2;
  }
  else
  {
          if (listen(fd, MAX_CONNECTION_NUMBER) < 0)
          {
                rval =  -3;
          }
          else
          {
            return fd;
          }
  }
  int err;
  err = errno;
  close(fd);
  errno = err;
  return rval;
}

int unix_socket_accept(int listenfd, uid_t *uidptr)
{
   int clifd, len, rval;
   time_t staletime;
   struct sockaddr_un un;
   struct stat statbuf;
   len = sizeof(un);
   if ((clifd = accept(listenfd, (struct sockaddr *)&un, &len)) < 0)
   {
      return(-1);
   }
 /* obtain the client's uid from its calling address */
   len -= offsetof(struct sockaddr_un, sun_path);  /* len of pathname */
   un.sun_path[len] = 0; /* null terminate */
   if (stat(un.sun_path, &statbuf) < 0)
   {
      rval = -2;
   }
   else
   {
           if (S_ISSOCK(statbuf.st_mode) )
           {
                  if (uidptr != NULL) *uidptr = statbuf.st_uid;    /* return uid of caller */
              unlink(un.sun_path);       /* we're done with pathname now */
                  return clifd;
           }
           else
           {
              rval = -3;     /* not a socket */
           }
    }
   int err;
   err = errno;
   close(clifd);
   errno = err;
   return(rval);
 }

int unix_socket_conn(const char *servername)
{ 
  int fd; 
  if ((fd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0)    /* create a UNIX domain stream socket */ 
  {
    return(-1);
  }
  int len, rval;
   struct sockaddr_un un;          
  memset(&un, 0, sizeof(un));            /* fill socket address structure with our address */
  un.sun_family = AF_UNIX; 
  sprintf(un.sun_path, "scktmp%05d", getpid()); 
  len = offsetof(struct sockaddr_un, sun_path) + strlen(un.sun_path);
  unlink(un.sun_path);               /* in case it already exists */ 
  if (bind(fd, (struct sockaddr *)&un, len) < 0)
  { 
  	 rval=  -2; 
  } 
  else
  {
	/* fill socket address structure with server's address */
	  memset(&un, 0, sizeof(un)); 
	  un.sun_family = AF_UNIX; 
	  strcpy(un.sun_path, servername); 
	  len = offsetof(struct sockaddr_un, sun_path) + strlen(servername); 
	  if (connect(fd, (struct sockaddr *)&un, len) < 0) 
	  {
		  rval= -4; 
	  } 
	  else
	  {
	     return (fd);
	  }
  }
  int err;
  err = errno;
  close(fd); 
  errno = err;
  return rval;	  
}
 
 void unix_socket_close(int fd)
 {
    close(fd);     
 }