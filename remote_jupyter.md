You can setup Jupyter in way so you can connect to it remotely 
from any computer (e.g. from your home). It can be very convenient 
as you don't need to go to the lab to work on a project remotely.

Since the lab machines are not accessible from the outside network, 
you need to first connect to the cs server (cs.uml.edu) and then connect 
to a lab machine (i.e. dan417-01.uml.edu). Moreover, you need to 
forward the Jupyter port (8888) to your local machine. 

Below is an instruction how to achieve this on a Mac or Linux system. 
Windows users can do the same using [putty](https://putty.org/).

Essentially, we need edit the ssh client config file, 
located in your home dir: `~/.ssh/config`. 
This file contains configurations options that are going to be used 
while connecting to a specific server. Note that this is a file on your local machine (your laptop).

Open this file in your favourite editor (or create it if it does not exists) and type the following:
```
Host cs
    HostName cs.uml.edu
    User your_username
``` 
where `your_username` is your actual user name from the cs server. Save the changes and close the file. 
Now, if you type `ssh cs` in the terminal, the ssh client understand that the hostname should be `cs.uml.edu`, 
and the username should be `your_username`. Similarly, you can specify other connection options 
under the `Host cs` directive.

Next, we will specify the port forwarding option, as well as the proxy connection using the cs server.
Open the `~/.ssh/config` file again and insert the code below at the end of the file:
```
Host dan417-01
    Hostname dan417-01.uml.edu
    User your_username
    LocalForward 8888 127.0.0.1:8888
    ProxyJump cs
```

The `LowelForward` option specifies that the connections to the port `8888` on your local machine should be
forwarded to the address `127.0.0.1:8888` on the remote machine. Since the Jupyter listens on this address by default, 
if you open the browser on your local machine and go to http://127.0.0.1:8888, you will connect to the Jupyter 
running on the remote server. 
