# 操作PDF

PDF的基本操作主要是读取、创建，合并等操作。使用Python的第三方包`PyPDF2`.

## 安装依赖包

使用Python的`pip`进行安装，安装包名称大小写不敏感。

```
pip install PyPDF2
```

### 查看基本的类和方法

进入包的`__init__.py`文件可以看到主要的几个类和方法

```python
from .pdf import PdfFileReader, PdfFileWriter
from .merger import PdfFileMerger
from .pagerange import PageRange, parse_filename_page_ranges
from ._version import __version__
__all__ = ["pdf", "PdfFileMerger"]
```

从名称上可以看出提供了基本的操作方法，分别是读取，写入和合并，其中合并可以认为是读取然后 再写入的操作。

## 读取和写入PDF

读取PDF非常简单，直接使用`PdfFileReader`这个类，先来看看这个类的参数

```python
class PdfFileReader(object):
    """
    Initializes a PdfFileReader object.  This operation can take some time, as
    the PDF stream's cross-reference tables are read into memory.

    :param stream: A File object or an object that supports the standard read
        and seek methods similar to a File object. Could also be a
        string representing a path to a PDF file.
    :param bool strict: Determines whether user should be warned of all
        problems and also causes some correctable problems to be fatal.
        Defaults to ``True``.
    :param warndest: Destination for logging warnings (defaults to
        ``sys.stderr``).
    :param bool overwriteWarnings: Determines whether to override Python's
        ``warnings.py`` module with a custom implementation (defaults to
        ``True``).
    """
    def __init__(self, stream, strict=True, warndest = None, overwriteWarnings = True):
```

其中必须传入的参数是`stream`，文件流而不是文件名称。 而PDF的创建不需要传入参数。

```python
from PyPDF2 import PdfFileReader, PdfFileWriter
infn = 'infn.pdf'
outfn = 'outfn.pdf'
# 获取一个 PdfFileReader 对象
pdf_input = PdfFileReader(open(infn, 'rb'))
# 获取 PDF 的页数
page_count = pdf_input.getNumPages()
print(page_count)
# 返回一个 PageObject
page = pdf_input.getPage(i)

# 获取一个 PdfFileWriter 对象
pdf_output = PdfFileWriter()
# 将一个 PageObject 加入到 PdfFileWriter 中
pdf_output.addPage(page)
# 输出到文件中
pdf_output.write(open(outfn, 'wb'))
```

## 合并多个PDF

合并多个PDF就读取多个文件，然后写入一个文件中。不过这样需要计算页面数，不如直接用`PdfFileMerger`,提供了`append`方法

```python
from PyPDF2 import PdfFileReader, PdfFileMerger


def read_pdf(pdf_name):
    stream = open(pdf_name, "rb")
    reader = None
    try:
        reader = PdfFileReader(stream)
    except Exception as e:
        print(e)
    return reader


def merge_pdf(pdfs, output_name = "merge.pdf"):
    merge = PdfFileMerger()
    for pdf_name in pdfs:
        pdf_obj = read_pdf(pdf_name)
        print("开始合并 《%s》 页面数: %s" % (pdf_name, pdf_obj.getNumPages()))
        merge.append(pdf_obj)
    merge.write(open(output_name, "wb"))
    print("合并后总页面数：", merge.id_count)
    print("写入当前目录", output_name)


if __name__ == '__main__':
    pdfs = [
        "file1.pdf", "file2.pdf", "file3.pdf"
    ]
    merge_pdf(pdfs)
```

输出结果：

```
开始合并 《file1.pdf》 页面数: 28
开始合并 《file2.pdf》 页面数: 34
开始合并 《file3.pdf》 页面数: 38
合并后总页面数： 100
写入当前目录 merge.pdf
```
