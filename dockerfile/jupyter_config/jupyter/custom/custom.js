require(["nbextensions/snippets_menu/main"], function (snippets_menu) {
    console.log('Loading `snippets_menu` customizations from `custom.js`');

    var magics = {//{{{
        'name': 'Magics',
        'sub-menu-direction': 'left',
        'sub-menu' : [
            {
                'name': '_IMPORT',//{{{
                'snippet' : [
                    "%reload_ext watermark",
                    "%reload_ext autoreload",
                    "%autoreload 2",
                    "# %watermark -p numpy,sklearn,pandas",
                    "# %watermark -p ipywidgets,cv2,PIL,matplotlib,plotly,netron",
                    "# %watermark -p torch,torchvision,torchaudio",
                    "# %watermark -p tensorflow,tensorboard,tflite",
                    "# %watermark -p onnx,tf2onnx,onnxruntime,tensorrt,tvm",
                    "# %matplotlib inline",
                    "# %config InlineBackend.figure_format='retina'",
                    "# %config IPCompleter.use_jedi = False",
                    "",
                    "%matplotlib inline",
                    "# %matplotlib widget",
                    "# from IPython.display import display, Markdown, HTML, IFrame, Image, Javascript",
                    "# from IPython.core.magic import register_line_cell_magic, register_line_magic, register_cell_magic",
                    "# display(HTML('<style>.container { width:%d%% !important; }</style>' % 90))",
                    "",
                    "import sys, os, io, logging, time, random, math",
                    "import json, base64, requests, shutil",
                    "import argparse, shlex, signal",
                    "import numpy as np",
                    "",
                    "argparse.ArgumentParser.exit = lambda *arg, **kwargs: _IGNORE_",
                    "",
                    "def _IMPORT(x, tag='main', debug=False):",
                    "    def __request_text(url):",
                    "        response = requests.get(url)",
                    "        if response.status_code == 200:",
                    "            return response.text",
                    "        else:",
                    "            raise RuntimeError(url)",
                    "    try:",
                    "        x = x.strip()",
                    "        if x[0] == '/' or x[1] == '/':",
                    "            with open(x) as fr:",
                    "                x = fr.read()",
                    "        elif 'github' in x or 'gitee' in x:",
                    "            if x.startswith('import '):",
                    "                x = x[7:]",
                    "            if x.startswith('https://'):",
                    "                x = x[8:]",
                    "            if not x.endswith('.py'):",
                    "                x = x + '.py'",
                    "            x = x.replace('blob/main/', '').replace('blob/master/', '')",
                    "            if x.startswith('raw.githubusercontent.com'):",
                    "                x = 'https://' + x",
                    "                x = __request_text(x)",
                    "            elif x.startswith('github.com'):",
                    "                x = x.replace('github.com', 'raw.githubusercontent.com')",
                    "                mod = x.split('/')",
                    "                x = 'https://' + '/'.join(mod[:3]) + f'/{tag}/' + '/'.join(mod[-3:])",
                    "                x = __request_text(x)",
                    "            elif x.startswith('gitee.com'):",
                    "                mod = x.split('/')",
                    "                x = 'https://' + '/'.join(mod[:3]) + f'/raw/{tag}/' + '/'.join(mod[3:])",
                    "                x = __request_text(x)",
                    "        if debug:",
                    "            return x",
                    "        else:",
                    "            exec(x, globals())",
                    "    except Exception as err:",
                    "        # sys.stderr.write(f'request {x} : {err}')",
                    "       pass",
                    "",
                    "def _DIR(x, dumps=True, ret=True):",
                    "    attrs = sorted([y for y in dir(x) if not y.startswith('_')])",
                    "    result = '%s: %s' % (str(type(x))[8:-2], json.dumps(attrs) if dumps else attrs)",
                    "    if ret:",
                    "        return result",
                    "    print(result)",
                    "",
                ]
            },//}}}
            '---',
            {
                'name': 'Custom Magics',//{{{
                'sub-menu': [
                    {
                        'name': 'Template WriteFile',//{{{
                        'snippet': [
                            "",
                            "@register_line_cell_magic",
                            "def template_writefile(line, cell):",
                            "    path = os.path.dirname(line)",
                            "    if not os.path.exists(path):",
                            "        os.makedirs(path, exist_ok=True)",
                            "    with open(line, 'w') as fw:",
                            "        fw.write(cell.format(**globals()))",
                            "",
                        ]
                    },//}}}
                    {
                        'name': 'Kill Process', //{{{
                        'snippet': [
                            "",
                            "@register_line_magic",
                            "def killall(line):",
                            "    pid_list=!ps -eo pid,command | grep $line  | grep -v 'grep' | cut -c1-6",
                            "    for pid in pid_list:",
                            "        pid = pid.strip()",
                            "        !kill -9 $pid",
                            ""
                        ]
                    },//}}}
                    '---',
                    {
                        'name': 'Html Display(*)',//{{{
                        'snippet': [
                            "",
                            "def display_html(port, height=600):",
                            "    from IPython import display",
                            "    from html import escape as html_escape",
                            "    frame_id = 'erlangai-frame-{:08x}'.format(random.getrandbits(64))",
                            "    shell = '''",
                            "      <iframe id='%HTML_ID%' width='100%' height='%HEIGHT%' frameborder='0'>",
                            "      </iframe>",
                            "      <script>",
                            "        (function() {",
                            "          const frame = document.getElementById(%JSON_ID%);",
                            "          const url = new URL(%URL%, window.location);",
                            "          const port = %PORT%;",
                            "          if (port) {",
                            "            url.port = port;",
                            "          }",
                            "          frame.src = url;",
                            "        })();",
                            "      </script>",
                            "    '''",
                            "    replacements = [",
                            "        ('%HTML_ID%', html_escape(frame_id, quote=True)),",
                            "        ('%JSON_ID%', json.dumps(frame_id)),",
                            "        ('%HEIGHT%', '%d' % height),",
                            "        ('%PORT%', '%d' % port),",
                            "        ('%URL%', json.dumps('/')),",
                            "    ]",
                            "    for (k, v) in replacements:",
                            "        shell = shell.replace(k, v)",
                            "    display.display(display.HTML(shell))",
                            "",
                        ],
                        'sub-menu': [
                            {
                                'name': 'Netron Display',//{{{
                                'snippet': [
                                    "",
                                    "@register_line_magic",
                                    "def netron(line):",
                                    "    parser = argparse.ArgumentParser(prog='netron')",
                                    "    parser.add_argument('--file', '-f', type=str, required=True, help='netron model file')",
                                    "    parser.add_argument('--port', '-p', type=int, default=0, help='netron server port')",
                                    "    parser.add_argument('--height', type=int, default=500, help='display netron html window hight')",
                                    "    import netron",
                                    "    try:",
                                    "        args = parser.parse_args(shlex.split(line))",
                                    "        address = netron.start(args.file, address=('0.0.0.0', args.port), browse=False)",
                                    "        display_html(address[1], args.height)",
                                    "    except:",
                                    "        pass",
                                    "",
                                ],
                            }, //}}}
                            {
                                'name': 'Tensorboard Display',//{{{
                                'snippet': [
                                    "",
                                    "@register_line_magic",
                                    "def tensorboard(line):",
                                    "    parser = argparse.ArgumentParser(prog='tensorboard')",
                                    "    parser.add_argument('--logdir', '-d', type=str, required=True, help='tensorboard logdir')",
                                    "    parser.add_argument('--port', '-p', type=int, required=True, help='tensorboard server port')",
                                    "    parser.add_argument('--height', type=int, default=600, help='display netron html window hight')",
                                    "    from tensorboard import manager as tbmanager",
                                    "",
                                    "    infos = tbmanager.get_all()",
                                    "    for info in infos:",
                                    "        if info.port != port: continue",
                                    "        try:",
                                    "            os.kill(info.pid, signal.SIGKILL)",
                                    "            os.unlink(os.path.join(tbmanager._get_info_dir(), f'pid-{info.pid}.info'))",
                                    "        except OSError as e:",
                                    "            if e.errno != errno.ENOENT: raise",
                                    "        except Exception:",
                                    "            pass",
                                    "        break",
                                    "",
                                    "    try:",
                                    "        args = parser.parse_args(shlex.split(line))",
                                    "        strargs = f'--host 0.0.0.0 --port {args.port} --logdir {args.logdir} --reload_interval 10'",
                                    "        command = shlex.split(strargs, comments=True, posix=True)",
                                    "        tbmanager.start(command)",
                                    "        display_html(args.port, args.height)",
                                    "    except:",
                                    "        pass",
                                    ""
                                ]
                            },//}}}
                        ]
                    },//}}}
                ]
            },//}}}
        ]
    };//}}}

    var utils = {//{{{
        'name': 'Utils',
        'sub-menu-direction': 'left',
        'sub-menu': [
            {
                'name': 'Common',
                'sub-menu': [
                    {
                        'name': 'Logging',
                        'snippet': [
                            "",
                            "def get_logger(name, level=logging.DEBUG, filepath=None, console=True):",
                            "    logger = logging.getLogger(name)",
                            "    logger.setLevel(level)",
                            "    #  %(filename)s",
                            "    formatter = logging.Formatter('%(asctime)s - %(funcName)s:%(lineno)d - %(name)s - %(levelname)s - %(message)s')",
                            "    if console:",
                            "        console = logging.StreamHandler()",
                            "        console.setLevel(level)",
                            "        console.setFormatter(formatter)",
                            "        logger.addHandler(console)",
                            "    if filepath:",
                            "        filelog = logging.FileHandler(filepath)",
                            "        filelog.setLevel(level)",
                            "        filelog.setFormatter(formatter)",
                            "        logger.addHandler(filelog)",
                            "    return logger",
                            ""
                        ]
                    },
                    {
                        'name': 'Json Encoder',
                        'snippet': [
                            "",
                            "import json, functools, datetime",
                            "import numpy as np",
                            "",
                            "class __JsonEncoder(json.JSONEncoder):",
                            "    def default(self, obj):",
                            "        if isinstance(obj, (datetime.datetime, datetime.timedelta)):",
                            "            return '{}'.format(obj)",
                            "        elif isinstance(obj, np.integer):",
                            "            return int(obj)",
                            "        elif isinstance(obj, np.floating):",
                            "            return float(obj)",
                            "        elif isinstance(obj, np.ndarray):",
                            "            return obj.tolist()",
                            "        else:",
                            "            return json.JSONEncoder.default(self, obj)",
                            "",
                            "json.dumps = functools.partial(json.dumps, cls=__JsonEncoder)",
                            ""
                        ]
                    },
                    {
                        'name': 'Print Progress Bar',//{{{
                        'snippet': [
                            "",
                            "from tqdm.notebook import tqdm",
                            "def print_progress_bar(x):",
                            "    print('\\r', end='')",
                            "    print('Progress: {}%:'.format(x), '%s%s' % ('▋'*(x//2), '.'*((100-x)//2)), end='')",
                            "    sys.stdout.flush()",
                            ""
                        ]
                    },//}}}
                    {
                        'name': 'Random Seed',//{{{
                        'snippet': [
                            "",
                            "def  set_rng_seed(x):",
                            "    try:",
                            "        random.seed(x)",
                            "        np.random.seed(x)",
                            "        torch.manual_seed(x)",
                            "    except: ",
                            "        pass",
                            "",
                            "set_rng_seed(888)",
                            ""
                        ]
                    },//}}}
                    {
                        'name': 'Image to Base64',//{{{
                        'snippet': [
                            "",
                            "def img2bytes(x, width=None, height=None):",
                            "    if isinstance(x, bytes):",
                            "        return x",
                            "",
                            "    if isinstance(x, str):",
                            "        if os.path.isfile(x):",
                            "            x = PIL.Image.open(x).convert('RGB')",
                            "        else:",
                            "            import cairosvg",
                            "            with io.BytesIO() as fw:",
                            "                cairosvg.svg2png(bytestring=x, write_to=fw,",
                            "                        output_width=width, output_height=height)",
                            "                return fw.getvalue()",
                            "",
                            "    from matplotlib.figure import Figure",
                            "    if isinstance(x, Figure):",
                            "        with io.BytesIO() as fw:",
                            "            plt.savefig(fw)",
                            "            return fw.getvalue()",
                            "",
                            "    from torch import Tensor",
                            "    from torchvision import transforms",
                            "    from PIL import Image",
                            "    if isinstance(x, Tensor):",
                            "        x = transforms.ToPILImage()(x)",
                            "    elif isinstance(x, np.ndarray):",
                            "        x = Image.fromarray(x.astype('uint8')).convert('RGB')",
                            "",
                            "    if isinstance(x, Image.Image):",
                            "        if width and height:",
                            "            x = x.resize((width, height))",
                            "        with io.BytesIO() as fw:",
                            "            x.save(fw, format='PNG')",
                            "            return fw.getvalue()",
                            "    raise NotImplementedError(type(x))",
                            "",
                            "def img2b64(x):",
                            "    return base64.b64encode(img2bytes(x)).decode()",
                            ""
                        ]
                    },//}}}
                    {
                        'name': 'Color Define', ///{{{{
                        'snippet': [
                            "",
                            "class COLORS(object):",
                            "    # BGR",
                            "    GREEN      = (0   , 255 , 0)",
                            "    RED        = (0   , 0   , 255)",
                            "    BLACK      = (0   , 0   , 0)",
                            "    YELLOW     = (0   , 255 , 255)",
                            "    WHITE      = (255 , 255 , 255)",
                            "    CYAN       = (255 , 255 , 0)",
                            "    MAGENTA    = (255 , 0   , 242)",
                            "    GOLDEN     = (32  , 218 , 165)",
                            "    LIGHT_BLUE = (255 , 9   , 2)",
                            "    PURPLE     = (128 , 0   , 128)",
                            "    CHOCOLATE  = (30  , 105 , 210)",
                            "    PINK       = (147 , 20  , 255)",
                            "    ORANGE     = (0   , 69  , 255)",
                            "",
                        ]
                    },///}}}
                ]
            },
            '---',
            {
                'name': 'Display Function(*)',//{{{
                'snippet': [
                        "",
                        "###",
                        "### Display ###",
                        "###",
                        "",
                        "_IMPORT('import pandas as pd')",
                        "_IMPORT('import cv2')",
                        "_IMPORT('from PIL import Image')",
                        "_IMPORT('import matplotlib.pyplot as plt')",
                        "_IMPORT('import plotly')",
                        "_IMPORT('import plotly.graph_objects as go')",
                        "_IMPORT('import ipywidgets as widgets')",
                        "_IMPORT('from ipywidgets import interact, interactive, fixed, interact_manual')",
                        "",
                        "# plotly.offline.init_notebook_mode(connected=False)",
                        "",
                        "plt.rcParams['figure.figsize'] = (12.0, 8.0)",
                        "# from matplotlib.font_manager import FontProperties",
                        "# simsun = FontProperties(fname='/sysfonts/simsun.ttc', size=12)",
                        "",
                ],
                'sub-menu': [
                    {
                        'name': 'Image Read',//{{{
                        'snippet': [
                            "",
                            "def im_read(url, rgb=True, size=None):",
                            "    if url.startswith('http'):",
                            "        response = requests.get(url)",
                            "        if response:",
                            "            imgmat = np.frombuffer(response.content, dtype=np.uint8)",
                            "            img = cv2.imdecode(imgmat, cv2.IMREAD_COLOR)",
                            "        else:",
                            "            return None",
                            "    else:",
                            "        img = cv2.imread(url)",
                            "        ",
                            "    if rgb:",
                            "        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)",
                            "    if size:",
                            "        if isinstance(size, int):",
                            "            size = (size, size)",
                            "        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)",
                            "    return img",
                            "",
                        ]
                    },//}}}
                    {
                        'name': 'Show Table(MD)',//{{{
                        'snippet': [
                            "",
                            "def show_table(headers, data, width=900):",
                            "    from IPython.display import Markdown",
                            "    ncols = len(headers)",
                            "    width = int(width / ncols)",
                            "    lralign = []",
                            "    caption = []",
                            "    for item in headers:",
                            "        astr = ''",
                            "        if item[0] == ':':",
                            "            astr = ':'",
                            "            item = item[1:]",
                            "        astr += '---'",
                            "        if item[-1] == ':':",
                            "            astr += ':'",
                            "            item = item[:-1]",
                            "        lralign.append(astr)",
                            "        caption.append(item)",
                            "    captionstr = '|'.join(caption) + chr(10)",
                            "    lralignstr = '|'.join(lralign) + chr(10)",
                            "    imgholdstr = '|'.join(['<img width=%d/>' % width] * ncols) + chr(10)",
                            "    table = captionstr + lralignstr + imgholdstr",
                            "    is_dict = isinstance(data[0], dict)",
                            "    for row in data:",
                            "        if is_dict:",
                            "            table += '|'.join([f'{row[c]}' for c in caption]) + chr(10)",
                            "        else:",
                            "            table += '|'.join([f'{col}' for col in row]) + chr(10)",
                            "    return Markdown(table)",
                            "",
                        ]
                    },//}}}
                    {
                        'name': 'Show Video',//{{{
                        'snippet': [
                            "",
                            "def show_video(vidsrc, width=None, height=None):",
                            "    W, H = '', ''",
                            "    if width:",
                            "        W = 'width=%d' % width",
                            "    if height:",
                            "        H = 'height=%d' % height",
                            "    if vidsrc.startswith('http'):",
                            "        data_url = vidsrc",
                            "    else:",
                            "        mp4 = open(vidsrc, 'rb').read()",
                            "        data_url = 'data:video/mp4;base64,' + base64.b64encode(mp4).decode()",
                            "    return HTML('<center><video %s %s controls src=\"%s\" type=\"video/mp4\"/></center>' % (W, H, data_url))",
                            "",
                        ]
                    },//}}}
                    {
                        'name': 'Show Image',//{{{
                        'snippet': [
                            "",
                            "def show_image(imgsrc, width=None, height=None, rgb=True):",
                            "    if isinstance(imgsrc, np.ndarray):",
                            "        img = imgsrc",
                            "        if width or height:",
                            "            if width and height:",
                            "                size = (width, height)",
                            "            else:",
                            "                rate = img.shape[1] / img.shape[0]",
                            "                if width:",
                            "                    size = (width, int(width/rate))",
                            "                else:",
                            "                    size = (int(height*rate), height)",
                            "            img = cv2.resize(img, size)",
                            "            plt.figure(figsize=(3*int(size[0]/80+1), 3*int(size[1]/80+1)), dpi=80)",
                            "        plt.axis('off')",
                            "        if len(img.shape) > 2:",
                            "            if not rgb:",
                            "                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)",
                            "            plt.imshow(img);",
                            "        else:",
                            "            plt.imshow(img, cmap='gray');",
                            "        return",
                            "",
                            "    W, H = '', ''",
                            "    if width:",
                            "        W = 'width=%d' % width",
                            "    if height:",
                            "        H = 'height=%d' % height",
                            "    if imgsrc.startswith('http'):",
                            "        data_url = imgsrc",
                            "    else:",
                            "        if len(imgsrc) > 2048:",
                            "            data_url = 'data:image/jpg;base64,' + imgsrc",
                            "        else:",
                            "            img = open(imgsrc, 'rb').read()",
                            "            data_url = 'data:image/jpg;base64,' + base64.b64encode(img).decode()",
                            "    return HTML('<center><img %s %s src=\"%s\"/></center>' % (W, H, data_url))",
                            "",
                        ]
                    },//}}}
                    {
                        'name': 'NBEasy Display',//{{{
                        'snippet': [
                            "",
                            "_IMPORT('gitee.com/qrsforever/nb_easy/easy_widget')",
                            "",
                            "def nbeasy_widget_display(images, img_wid=None):",
                            "    if isinstance(images, np.ndarray):",
                            "        images = {'_': images}",
                            "    elif isinstance(images, tuple) or isinstance(images, list):",
                            "        images = {f'_{i}': img for i, img in enumerate(images)}",
                            "    C = len(images)",
                            "    if C == 0:",
                            "        return None",
                            "    show_ncol, show_nrow = 1, 1",
                            "    if C > 1:",
                            "        if img_wid:",
                            "            show_ncol = 2 if int(img_wid.width) <= 1300 else 1",
                            "        for i in range(C % show_ncol):",
                            "            images[f'placehold-{i}'] = images[list(images.keys())[-1]].copy()",
                            "        show_nrow = len(images) // show_ncol",
                            "        row_images = []",
                            "        col_images = []",
                            "        for key, img in images.items():",
                            "            if not key.startswith('_'):",
                            "                cv2.putText(img, key, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 6, 2), 1)",
                            "            col_images.append(img)",
                            "            if len(col_images) == show_ncol:",
                            "                row_images.append(np.hstack(col_images))",
                            "                col_images = []",
                            "",
                            "        display_image = np.vstack(row_images)",
                            "    else:",
                            "        display_image = images.popitem()[1]",
                            "    ",
                            "    if img_wid:",
                            "        img_wid.layout.width = f'{display_image.shape[1]}px'",
                            "        img_wid.layout.height = f'{display_image.shape[0]}px'",
                            "        if isinstance(img_wid, widgets.Image):",
                            "            img_wid.value = io.BytesIO(cv2.imencode('.png', display_image)[1]).getvalue()",
                            "        else:",
                            "            img_wid.height, img_wid.width = display_image.shape[0], display_image.shape[1]",
                            "            img_wid.put_image_data(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))",
                            "            img_wid.send_state()",
                            "    else:",
                            "        return display_image",
                            "",
                        ]
                    }, //}}}
                ],
            },//}}}
        ]
    };//}}}

    var erlangai = {//{{{
        'name' : 'ErlangAI',
        'sub-menu-direction': 'left',
        'sub-menu' : [
            {
                'name': 'Pytorch(*)',
                'snippet': [
                    "",
                    "###",
                    "### Torch ###",
                    "###",
                    "",
                    "import torch",
                    "import torch.nn as nn",
                    "import torch.nn.functional as F",
                    "import torch.optim as O",
                    "from torchvision import models as M",
                    "from torchvision import transforms as T",
                    "from torch.utils.data import Dataset, DataLoader",
                    ""
                ],
                'sub-menu': [
                    {
                        'name': 'Uknow',
                        'snippet': [
                        ],
                    }
                ]
            },
            {
                'name': 'Tensorflow(*)',//{{{
                'snippet': [
                    "",
                    "###",
                    "### Tensorflow ###",
                    "###",
                    "",
                    "import tensorflow as tf",
                    "import tf2onnx",
                    "",
                ]
            },//}}}
            '---',
            {
                'name': 'Onnx(*)',//{{{
                'snippet': [
                    "",
                    "###",
                    "### Onnx ###",
                    "###",
                    "",
                    "import onnx",
                    "import onnx.helper as OH",
                    "import onnxruntime as rt",
                    "",
                ]
            }//}}}
        ],
    };//}}}

    var markdown = {//{{{
        'name' : 'Markdown',
        'sub-menu-direction': 'left',
        'sub-menu' : [
            {
                'name': 'Alert',//{{{
                'sub-menu': [
                    {
                        'name': 'Info',
                        'snippet': [
                            '<div class="alert alert-info">',
                            '',
                            '</div>',
                        ]
                    },
                    {
                        'name': 'Warning',
                        'snippet': [
                            '<div class="alert alert-warning">',
                            '',
                            '</div>',
                        ]
                    },
                    {
                        'name': 'Danger',
                        'snippet': [
                            '<div class="alert alert-danger">',
                            '',
                            '</div>',
                        ]
                    },
                    {
                        'name': 'Success',
                        'snippet': [
                            '<div class="alert alert-success">',
                            '',
                            '</div>',
                        ]
                    },
                ]//}}}
            },
            {
                'name': 'Emoji(*)',//{{{
                'snippet': [
                    '<div style="margin-top:30px; width:100%;">',
                    '    <div style="float:left;font-size:18px;">',
                    '        &#128279;&#128285;&#9757;',
                    '    </div>',
                    '</div>',
                ],
                'sub-menu': [
                    {
                        'name': 'Link&Top',
                        'snippet': [
                            '<div style="margin-top:30px; width: 100%;">',
                            '    <div style="float:left;">',
                            '        <a',
                            '           title=""',
                            '           href=""',
                            '           style="color:blue;font-size:28px;text-decoration:none;">',
                            '                &#128279;',
                            '        </a>',
                            '    </div>',
                            '    <div style="text-align:right">',
                            '        <a',
                            '           title="Back to Top"',
                            '           href="#Contents"',
                            '           onclick="window.scrollTo(0, 0);"',
                            '           style="color:blue;font-size:28px;text-decoration:none;">',
                            '               &#128285;',
                            '        </a>',
                            '    </div>',
                            '</div>',
                        ]
                    },
                ]//}}}
            },
        ]
    };//}}}

    // snippets_menu.options['menus'].push(snippets_menu.default_menus[0]);
    snippets_menu.options['menus'].push(magics);
    snippets_menu.options['menus'].push(utils);
    snippets_menu.options['menus'].push(erlangai);
    snippets_menu.options['menus'].push(markdown);
    // console.log(snippets_menu)
    console.log('Loaded `snippets_menu` customizations from `custom.js`');
});

/////////////////////////////////////////////////////////////////////////////////////

require.undef('VideoEModel');

define('VideoEModel', ["@jupyter-widgets/controls"], function(widgets) {
    var VideoEView = widgets.VideoView.extend({
         update: function() {
             if (this.buttons == undefined) {
                 this.el.setAttribute('crossOrigin', 'anonymous');
                 this.on_snapshot_click = 0;
                 setTimeout(function(that) {
                     that.buttons = document.createElement('div');
                     that.buttons.classList.add('widget-inline-hbox');
                     that.buttons.classList.add('widget-play');
                     that.snapshot_btn = document.createElement('button');
                     that.snapshot_btn.className = 'jupyter-button';
                     const snapshot_icon = document.createElement('i');
                     snapshot_icon.className = 'fa fa-camera';
                     that.buttons.appendChild(that.snapshot_btn);
                     that.snapshot_btn.appendChild(snapshot_icon);
                     that.el.appendChild(that.buttons);
                     that.el.parentNode.insertBefore(that.buttons, that.el);
                     that.snapshot_btn.onclick = () => {
                         // debugger
                         that.on_snapshot_click = 2;
                         let canvas = document.createElement('canvas');
                         let canvas_ctx = canvas.getContext('2d');
                         canvas.width = that.el.videoWidth || 600;
                         canvas.height = that.el.videoHeight || 352;
                         canvas_ctx.drawImage(that.el, 0, 0, canvas.width, canvas.height);
                         that.model.set('imgb4str', canvas.toDataURL('image/png'));
                         that.model.save_changes();
                     };
                 }, 1000, this);
             }
             if (this.on_snapshot_click > 0) {
                 this.on_snapshot_click -= 1
             } else {
                 widgets.VideoView.prototype.update.call(this);
             }
         },
    });
    return {
        VideoEView : VideoEView,
    }
});
