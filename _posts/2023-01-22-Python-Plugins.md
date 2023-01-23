---
layout: post
title:  "Plugins in Python"
date:   2023-01-22 14:14:49 +0530
categories: python
---

### Introduction

There are many cases in which a library or an application might want to expose some customization such that developers can affect execution behavior. A great example of this would be `pytest` framework that allows external plug-ins to provide more features than the basic features that the  provies. Another example is supporting multiple compiler backends for PyTorch 2.0's `torch.compile` (See [issue](https://github.com/pytorch/pytorch/issues/91824#issuecomment-1386532225)).

#### Potential Solutions

##### Intrusive Changes

One approach is to update the codebase everytime a new extension is to be registered. However, this approach is not scalable and may become burdening for the maintainers.

##### Plug-in

A non-invasive approach will involve a plug-in capability so that the existing code is not affected everytime a new plugin is installed. For the plug-in approach to work, the main library has to publicly state what it expects of the plug-in or how it interacts with the plug-in interface. In this blog, we will dive deeper into a simple plugin example with Python.

#### Python Entry Point Specification

Before we dive into the implementation, let us understand a bit about Python's Entry Point Specification which will help us implement this with ease. From setuptools [documentation](https://setuptools.pypa.io/en/latest/userguide/entry_point.html), we understand that

> Entry points are a type of metadata that can be exposed by s on installation. 

It is useful when a package would like to enable customization of its functionalities via plugins.

We will be leveraging the entry point specification to implement a simple package which allows customizations via plug-in.

#### Example

##### Script to consume the plugins
In our simple example, we will have a script which will try to look for a set of plugins registered for `my-plugins` namespace.
It will then load those plug-ins and call `do_something` method on the loaded Python object.

**NOTE**: It is not required that plug-in points to an object. It could also point to a function or module.

```python
# pkg_resources will have us find and load the plug-in
import pkg_resources

def load_plugins():
    plugins = dict()
    # Find all plugins registered under `my-plugins`.
    for entry_point in pkg_resources.iter_entry_points('my-plugins'):
        # Load the plugin object and store it in a dictionary.
        plugins[entry_point.name] = entry_point.load()
    return plugins

plugins = load_plugins()

# Iterate over loaded plugins
# and call the `do_something` method.
for name, plugin in plugins.items():
    print(f"Executing plugin {name}")
    plugin.do_something()

```

For simplicity, our script is very simple which tries to find the plug-ins located under `my-plugins`.
We use the `pkg_resources` to locate the packages which have registered `my-plugins` as an entry-point.
Once the plugins are loaded, we iterate over the plugins and call `do_something` method on them. As long as
our plugins implement the expected interface (only the `do_something` method in our case), we can call them
as per our program.

##### Writing a plugin

The only thing our plugin needs to do is implement the expected interface (`do_something` method in our case).

```python
class PluginA:
    def __init__(self):
        pass

    def do_something(self):
        print("Plugin A is doing something")

plugin = PluginA()
```

There is nothing fancy happening here, it is good old Python code. The magic happens in `setup.py` for our plugin package.

```python
# setup.py
from setuptools import setup

setup(
    name='plugin_a',  # package name
    entry_points={  # you can have multiple entry-points for a package
        'my-plugins': [
            'plugin_a = plugin_a:plugin',
        ],
    }
)

```

Once you install the plugin package and try running our script above, it will print

```bash
Executing plugin plugin_a
Plugin A is doing something
```

As we can see, our plugin is named as `plugin_a` as named in our `setup.py` and the `do_something` method of our plugin prints `Plugin A is doing something`.

We have implemented a very simple plugin example, but one can extend it to more complex plugin systems.

This code can be found on Github at this [link](https://github.com/kshitij12345/python-plugin)

#### Conclusion

We learned about
* why plugin system can be useful for a project
* how Python supports it with entry point specification
* an example of a simple plugin system

References:
A really great reference was this [blog](https://amir.rachum.com/blog/2017/07/28/python-entry-points/) which walks you through a similar system in a fun way.
