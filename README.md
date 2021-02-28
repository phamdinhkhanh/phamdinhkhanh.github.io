# 1. Why do i write this post?
## 1.1. Why write blog is good?

I learn one experience that is:
"The best way to learn something is teach them again to the other"

Actually, I am quite normal learner when i start writing my blog. The unique motivation lead me to patiently write blog is that I have plenty of spare time. And I think I have to do something to fill in empty box. And one plan come up in my mind, that is writing about AI blog. Thus, why do I decise to write blog? Let me list the pros and cons:

**Advantage**:
- When you learn a new things, your neural conections is temporary. Thus, Docking the knowdge make you remember it better.
- You improve creation, arrangement and diligence. They are the important factors lead to the success.
- You improve coherent-writing skill.
- You can't thoroughly understand anything as if you can't explain in receptive way to the others. I refer to Albert Einstein's code.
- Your coding skill improved through preparing code for your topic.
- You become more brave when do a new thing that you are afraid to do before.
- Your blog kill your spare time. Time is the substance of the life, wasting the time is wasting your meaning of life.
- Your blog partially contribute to AI community. You should believe that your things honorly donate to shape the future. It make yourself better, make your audiences better and make your country better.
- You gain the greate renown if your blog is useful and supportive many audiences.

**Disavantage**:
- It requires your time. You maybe trade-off between entertain or work. Entertainment is necesssary person's need, but if you start writing one topic a week, it consumes your time merely 5 hours and you still have time for entertainment.
- You affraid that your knownledge is not good, you are not healthy, you don't have time for family, for girl friend or game .... You must define what are pivotal disciplines of your life? You can both balance your life and reserve your time to write blog.
- You scare the naysayers. You own your life, so you shouldn't let naysayers saying affect to you.
- You can't build up a website, latex, .... I know it can be a truly interception. But in this topic, I will share you my tip how to build A blog on jekyl.
- You can lost your rank when github stop free-domain service. I think is also a risk, you should buy a backup `.com` domain.

At the begin, I have to face up with the above troubles like you face. But I motivate my-self to be positive thinking and focusing to the advantage instead of disavantage. That is the secrete help me win.

When the first topic launched, I feel pleasure a litle, that moment reinforces to me that I can do it.

## 1.2. I hope you also write blog

You know, writing blog actually is not quite hard things. The benefit you gain is much than loss.

My aim, at the first time when I write blog, it just simple discovers AI. And now, I go beyond, my blog is welcomed. I have a group, fanpage to share many people also learn the new things like me.

In my experience, to write a good blog require you must have two things: 

- a perseverance to clinch your-self with favorite AI topic you want to share.
- presentation ability: At first, your blog must have good UI expression. Second, Your latex, fontsize, header must be uniformly show up. 

The first one is reply on your volition. The second one what I share you in this topic.

# 2. Technical behind blog

The blog source code is available at [phamdinhkhanh.github.io](https://github.com/phamdinhkhanh/phamdinhkhanh.github.io). You can clone as a reference template to build your blog.

To have a fancy UI, you don't need to be a javascript expert. I think what you require to know is techniques such as: jekyll, html, css, bootstrap at fundamental level.

The most important is jekyll. That is very good website framework facilitate to you manage posts in date.

<img src="/assets/images/20210208_BuildBlog/pic1.png" class="gigantic"/>

The audience can look through timeline and pickup the topic that they interesting. Thanks to the machinelearningcoban of A.TiepVu, I ideally have the idea for this template. He is very excellent when writes a meticulous blog like that. The code and latex are very carefully and clearly presented beside the blog's content is well assembled. by the way, I sincerely thank him so much.

## 2.1. jekyll

Jekyll is perfect framework for blog website. It able to transform your plain text into static website and blog easy. Further, It can be deployable on free domain like github. Writing AI blog is not as complicated as when you write dynamic website. Particularly, It does not requires API services, strong server, large storage database behind. So, static website on jekyll is quite fit to your needs.  

The typical structure of a jekyll blog that is settling many posts by date arrangement. You also keep layout of all posts in the same and change only content through embedding layout. Moreover, jekyll support you to seperate data with code by putting data in `_data` folder. You also store css style and images inside `_asset` folder. You can look at the representative tree directories of jekyll website to know more about it:

```
├── assets
│   ├── css
│   └── images
├── _includes
│   ├── home.html
│   └── navigation.html
├── index.html
├── _layouts
│   ├── default.html
│   └── post.html
├── _posts
│   ├── 2020-12-19-Resnet.md
│   ├── 2021-01-20-CycleGAN.md
│   ├── 2021-02-01-StarGAN.md
├── _sass
    ├── _main.scss
```

The meaning of each folder is avaible at the official tutorial [jekyll Step by Step Tutorial](https://jekyllrb.com/docs/step-by-step/01-setup/). Only following this tutorial in 2 hours, you can basically clarify how to build up a jekyll website? I summary the main contents in the tutorial as below:

* Some setup commands: 
    * install jekyll by gem
    * jekyll build
    * jekyll serve

* Posts: in `_post` folder. Storing all of your post. jekyll is very smart when auto setup the link for each post in `_post` folder. You can access several content of the post by link:
    * `post.url`: output path to your post.
    * `post.title`: title of your post.

* Layout: store in `_layout` folder. If your site include many repeated components, you should use layout to re-implement them. For example in my blog, I keep the top navigation bar, side menu is fix, only the blog content is varying. That is reason I should create default.html layout to apply them all in every my post.

* Include: store in `_include` folder. It is template files that you can bind into your page through `include` tag.

* Data Files: in `_data` folder. It also like hyperameter file where stores every config parameter for your site.

* Assets: in `_asset` folder. Where you save css (define style of html), images and javascript files.

* Liquid: Define what is going happen in the template of your website. Include three main components:
    * objects: It return value variable on page, bounded in curly brace `{{}}` , for example you can show your page title `{{ page.title }}`
    * tags: Used in logic flow, exspecially in `if` condition. That make your UI changes flexiable.
    * filter: Change the output of the object. It as the condition of output and separated by `|`. For example: `{{ "Hello World" | downcase}}` to make output is lowercase.
* Front matter: Define the variables for the page. That is snippet of YAML located between two triple-dashed lines and at the start of a file.

One very important part, To enable your site to show latex, you have to use supplement module. I trust [math-jax](http://www.iangoodfellow.com/blog/jekyll/markdown/tex/2016/11/07/latex-in-markdown.html) module because max-jax is recommended by renowned AI researcher Ian Goodfellow. Very simple, you just need to link it in the header of your `post.html` file and it is going to be valid at all posts.

I bet you that after watching through all the tutorials, you can easily build up your blog. But to decorate your blog more beautiful, you maybe learn about material design in addition. You can be able to perform your creation, arrangement, assembling-content skills to make your blog become glamorous and attractive. If you complete, kindly to share me your fruit.

## 2.2. domain github.io

After accomplishment of your website, you need to release it out on public domain. If you want your site to be your private seal, buying one domain is a good choice. If you write your site as a CV and don't care much about privacy, i think github.io is a very good option, and why is that?

**Advantage**:
- github.io is very good free domain, it have a high stability, low latency and high availability.
- easy to version control right in github.
- especially, It is free. You do not have to pay any cost for them annually.
- you known, many famous researcher also choose github.io as their blogs.
- you can publish your code to the community as contribution to motivate young bloggers.
- your repo can rated many stars and clone that give you renown in return.

**Disavantage**:
- you can not setup Google AdSense (I try and fail, if you are success, kindly let me know).
- many people think it is not professional and valueable.
- you want your repo must be private.

In my opinion, when I build the website, I consider a lot between two options: free-domain and non free-domain. But finally, I choice free-domain because my final target is share knownledge including the way to build my blog. If you keep your blog private, it is reversed the sharing spirit.

To see how to create a github page, you can look at very detail tutorial: [GithubPages](https://pages.github.com/). Github also stimulate you blogging your-self with jekyll and have a highlight tutorial [setting up a github pages with jekyll](https://docs.github.com/en/github/working-with-github-pages/setting-up-a-github-pages-site-with-jekyll).

# 3. The content

Write the content is not a easy job. Your blog is attractive or not mainly distributed by the content. Thus to have a fascinated content you should make a plan about what you are going to write. One good AI blog in my opinion it should be good at six traits:

- **Series post**: The posts should be gathered in specific domain and follow the sequence. Such as when you write about CNN models, you should introduce the concept of Convolutional, Receptive Field first. After that, you talk about the CNN backbones model according to the developement trend: LeNet, ZFNet, AlexNet, InceptionNet, MobileNet, ResNet, DenseNet, EfficientDet,.... It aims to audience remember better.   

- **Coherence**: Coherence meaning that the following sectors is related to previous sectors about content. If you write a post like you are telling a story for audience, your post will not be bored. It also drastically stimulates the curiosity of readers.

- **Having code**: You should set the role as you are audience. The expectation of audience when study your blog is not only know how the concept of algorithms but also how to practice them. Thus, your blog should be attached codes to be more facilitative and intituative.

- **Having Graph**: Although draw a picture is time wasting. But at least it make your blog become easily understanding to trivial audience. You also have the brief note bellow each picture to explain about the picture's content.

- **Spliting content**: Why should we split the post into many level sections? That is my experience after I read several outstanding AI books such as: [Machine learning yearning](https://www.dbooks.org/machine-learning-yearning-1501/), [Dive into Deep Learning](https://d2l.ai/), [Deep Learning](https://www.deeplearningbook.org/). The common points i see that they split them in many chapter. In each chapter they always split long article into small and detail chapters as possible. It make the audience easy to observe the knownledge without confusing.

- **Reference**: You should list down the reference book or article at the last, it helps to audience can study if they want to read more. Moreover, sometime you also forget your knownledge and the reference links maybe useful.

What is your opinion about this matter? Let me know if you want to add.
