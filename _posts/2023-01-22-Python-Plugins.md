---
layout: post
title:  "Plugins in Python"
date:   2023-01-22 14:14:49 +0530
categories: python
---

### Introduction

There are many cases in which a library or an application might want to expose some customization/extension such that users/developers can override/affect some execution behavior. A great example of this would be `pytest` framework allows external plug-ins to provide more features than the basic features that the package provies. Another example is supporting multiple compiler backends for PyTorch 2.0's `torch.compile` (See [issue](https://github.com/pytorch/pytorch/issues/91824#issuecomment-1386532225)).

#### Potential Solutions

##### Intrusive Changes

One approach is to update the codebase everytime a new extension is to be registered. However, this approach is not scalable and may become burdening for the maintainers.

##### Plug-in

A non-invasive approach will involve a plug-in capability so that the existing code is not affected everytime a new plugin is installed. For the plug-in approach to work, the main library/application has to publicly state what it expects of the plug-in or how it interacts with the plug-in interface. In this blog, we will dive deeper into a simple plugin example with Python.

#### Python Entry Point Specification

Before we dive into the implementation, let us understand a bit about Python's Entry Point Specification which will help us implement this with ease. From https://setuptools.pypa.io/en/latest/userguide/entry_point.html, we understand that

> Entry points are a type of metadata that can be exposed by packages on installation. 

It is useful when a package would like to enable customization of its functionalities via plugins.
