---
layout: post
title:  "Modern Effective C++: Chapter 2"
date:   2021-07-17 14:14:49 +0530
categories: c++
---
### Chapter 2 : `auto`

-----

Item 5: Prefer auto to explicit type declarations.

* Use `auto` whenever possible (helps you from explicitly adding the `types`)
* Also, it helps avoid few errors
   1. `auto` doesn't support uninitialized variables. Unlike `int x` where `x` is uninitialized, `auto x;` is not valid! It has to be either `auto x{1}` or `auto x = 1;` (1 is just some random value in this example)
   2. It helps to easily take get the value type of iterator objects. Replace `typename std::iterator_traits<It>::value_type
 currValue = *b;` -> `auto currValue = *b`
   3. For entities like lambda, their type is only known to compiler so how on earth can you make a variable for it using explicit type? `auto` to the rescue 😉 
* Also helps avoid subtle bugs (example: unsigned len = vec.size(), return type of vec.size() is `std::vector<T>::size_type`, however  since unsigned is platform dependent it can lead to bugs if it doesn't match `std::vector<T>::size_type`) (Refer example on page 40 which talks about vec.size() )
* Can avoid temporary object creation (Refer example on page 40 which talks about unordered map)
* For people worried about `auto` decreasing the readability should note that tons of succesful language have type inference where user doesn't need to explicitly specify the types. Also IDE and editor can/should help reveal the type.

-----

Item 6:  Use the explicitly typed initializer idiom when `auto` deduces undesired types.

* For some edge-cases, `auto` may deduce type which you don't actually want, in that case you should use the **explicitly type initializer** idiom. Eg. `auto x = static_cast<float>(get_double_eps())`, this approach is better than implicitly converting to float using `float x = get_double_eps()`. Also this will help you to get around the functions or interface which return `proxy` object (as `auto` will deduce the type to be proxy object and not the type it is acting as proxy for). Proxy objects are those which behave like a certain type but are not of that type. `std::vector::operator[]` returns proxy object (as reference to bits is not valid). (Refer to Item 6 in book for gory details 😄 )
* Prefer explicit cast over implicit cast!
