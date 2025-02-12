def generate_adress(city):
    if city=='Paris':
        return '35 Avenue Daumesnil, 94160, Saint-Mandé'
    elif city=='London':
        return '66 Weymouth Street, London, W1G 6NZ'
    elif city=='New York':
        return '271 Bleecker St, New York, NY 10014'

def generate_cover_letter(address, position, name, role, bank, date):
    cover_letter = r"""\documentclass[12pt]{article}

    \usepackage[french]{babel}
    \usepackage[top=2cm,bottom=2cm,left=2cm,right=2cm]{geometry}
    \usepackage{hyperref}
    \usepackage{graphicx}
    
    \newcommand{\indep}{\perp \!\!\! \perp}
    
    \makeatletter
    \renewcommand{\paragraph}{%
      \@startsection{paragraph}{4}%
      {\z@}{1.25ex \@plus 1ex \@minus .2ex}{-1em}%
      {\normalfont\normalsize}} % Remplacez "\bfseries" par rien pour supprimer le gras
    \makeatother
    
    \title{Cover Letter}
    \author{Lenny Zerbib}
    \date{\today}
    
    \usepackage{fancyhdr}
    \pagestyle{fancy}
    \fancyhead[L]{\textbf{""" + position + r"""}} 
    \fancyhead[C]{}
    \fancyhead[R]{\includegraphics[width = 0.3 \textwidth]{logodauphine.png}}
    
    
    \fancyfoot[L]{}
    \fancyfoot[C]{}
    \fancyfoot[R]{\textbf{""" + bank + r"""}} 
    \renewcommand{\footrulewidth}{0.4pt}
    
    \setlength{\headheight}{25.0485pt}
    \addtolength{\topmargin}{-13.0485pt}
    
    \begin{document}
    
    \begin{flushright}
        \href{https://www.linkedin.com/in/lenny-zerbib-b3569a235/}{Lenny Zerbib}\\
        {""" + address + r"""} \\
        +33 (0) 7 78 35 64 09 \\
        \href{mailto:lenny.zerbib@dauphine.eu}{lenny.zerbib@dauphine.eu}
    \end{flushright}
    
    \vspace{0.3cm}
    
    \paragraph{Dear {""" + name + r"""},}
    
    \vspace{0.3cm}
    
    \paragraph{\indent I am writing to express my sincere interest in applying for the 6-months position of {""" + role + r"""} Internship at {""" + bank + r"""} starting in {""" + date + r"""}. I am truly excited about the opportunity to present my portfolio for your esteemed consideration.}
    
    \vspace{0.3cm}
    
    \paragraph{As the valedictorian of a Master's Program in Applied Mathematics, where I specialized in Statistics, Stochastic Calculus, and Machine Learning, and having completed the Master 203, ranked \#6 master’s degree in Financial Markets worldwide, at Université Paris Dauphine-PSL, I am eager to immerse myself in the complexities of this position.}
    
    \vspace{0.3cm}
    
    \paragraph{I served as an Exotic Equity Derivatives Trader Intern within the Stocks Solution team at Natixis CIB. My role involved implementing various models to optimize the fitting of dividend yield trends observed over recent years, with the aim of minimizing the impact of epsilon on the P\&L of the book. Additionally, I calibrated parameters in the Local Stochastic Volatility model to achieve precise fits to the vol-of-vol and spot-vol correlation curves, utilizing Monte Carlo simulations. These experiences deepened my understanding of derivative trading strategies and refined my skills in financial modeling, contributing to my enthusiasm for pursuing a career in equity derivatives trading.}
    
    \vspace{0.3cm}
    
    \paragraph{During the global health crisis, I served as a Covid-19 Mediator, conducting vital antigen tests to curb the spread of the disease. This role, undertaken alongside my undergraduate studies in mathematics, revealed my organizational skills and unyielding commitment during high-pressure exam periods. Juggling the demands of academia with active participation in the pandemic response not only demonstrated my resilience but also honed my ability to prioritize, work efficiently, and consistently delivering exceptional performance.}
    
    \vspace{0.3cm}
    
    \paragraph{Thank you for considering my application. I am excited about the opportunity to be part of the Internship Program at {""" + bank + r"""}. I remain available in your time zone to proceed with the next steps at your earliest convenience.}
    
    \vspace{0.3cm}
    
    \paragraph{Yours sincerely,}
    
    \vspace{0.3cm}
    
    \paragraph{Lenny Zerbib}
    
    \end{document}"""
    
    return cover_letter