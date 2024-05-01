import{_ as t,r as a,o,c as r,a as s,b as p,d as e,e as l}from"./app-aAgoPGaO.js";const i="/assets/1-BC15midN.png",g="/assets/2-D84Hn7f_.png",h="/assets/3-Byl4fODP.png",c="/assets/4-0tdqFzVU.png",L="/assets/5-BI4vmml-.png",d="/assets/20-BEjeyFzX.png",m="/assets/6-BIAnzxjO.png",M="/assets/7-DP4y6rAj.png",_="/assets/8-Ctmwt4wt.png",f="/assets/9-BtwT9zU_.png",b="/assets/10-Dvh4LvqY.png",x="/assets/11-BModTLSR.png",u="/assets/15-rdPRLBKf.png",G="/assets/17-D5HMPHdy.png",B="/assets/18-CnXsEyr4.png",w="/assets/19-BN8PLOq4.png",T="/assets/16-BG_oD3zs.png",k="/assets/13-mCP8Yh-d.jpg",P="/assets/14-aeL58RtG.png",A="/assets/12-C96-X013.png",v={},C=s("h1",{id:"本地llm使用指南-0-3",tabindex:"-1"},[s("a",{class:"header-anchor",href:"#本地llm使用指南-0-3"},[s("span",null,"本地LLM使用指南 0.3")])],-1),U={href:"https://github.com/xue160709/Local-LLM-User-Guideline/blob/main/README-zh.md",target:"_blank",rel:"noopener noreferrer"},R=l('<p>作者：XueZhirong</p><h3 id="更新日期" tabindex="-1"><a class="header-anchor" href="#更新日期"><span>更新日期</span></a></h3><p><strong>2024-3-03</strong>：0.3版本增加了常用术语的介绍。</p><p><strong>2024-2-24</strong>：0.2版本增加了Gemma和Llava模型介绍，增加了GPT4ALL软件介绍，增加了LLM选型介绍。</p><p><strong>2024-2-17</strong>：0.1版本。</p><h2 id="_1-背景介绍" tabindex="-1"><a class="header-anchor" href="#_1-背景介绍"><span>1. 背景介绍</span></a></h2><h3 id="_1-1-什么是llms" tabindex="-1"><a class="header-anchor" href="#_1-1-什么是llms"><span>1.1 什么是LLMs？</span></a></h3><p>LLMs，即大型语言模型（Large Language Models），是一种基于人工智能和机器学习技术构建的先进模型，旨在理解和生成自然语言文本。这些模型通过分析和学习海量的文本数据，掌握语言的结构、语法、语义和上下文等复杂特性，从而能够执行各种语言相关的任务。LLM的能力包括但不限于文本生成、问答、文本摘要、翻译、情感分析等。</p><p>LLMs例如GPT、LLama、Mistral系列等，通过深度学习的技术架构，如Transformer，使得这些模型能够捕捉到文本之间深层次的关联和含义。模型首先在广泛的数据集上进行预训练，学习语言的一般特征和模式，然后可以针对特定的任务或领域进行微调，以提高其在特定应用中的表现。</p><p>预训练阶段让LLMs掌握了大量的语言知识和世界知识，而微调阶段则使模型能够在特定任务上达到更高的性能。这种训练方法赋予了LLMs在处理各种语言任务时的灵活性和适应性，能够为用户提供准确、多样化的信息和服务。</p><h3 id="_1-2-llms和应用程序的差异" tabindex="-1"><a class="header-anchor" href="#_1-2-llms和应用程序的差异"><span>1.2 LLMs和应用程序的差异</span></a></h3><h4 id="开放性" tabindex="-1"><a class="header-anchor" href="#开放性"><span>开放性</span></a></h4><p><strong>LLMs：</strong> 通过深度学习的预训练和微调过程，LLMs掌握了从海量互联网文本中获得的广泛知识，使其能够理解和生成包括文本、图片、音频、视频、3D素材在内的多种内容类型。这种能力让LLM在处理各种主题和知识领域时展现出卓越的多样化信息处理能力。</p><p><strong>应用程序：</strong> 应用程序则是为满足特定需求而设计，如社交互动、新闻获取、电子商务等，其开放性和内容多样性通常不如LLMs。每个应用都围绕其核心功能构建，提供用户界面和体验优化，但用户访问的信息和服务范围受限于应用的设计目的和功能界定。</p><h4 id="准确性" tabindex="-1"><a class="header-anchor" href="#准确性"><span>准确性</span></a></h4><p><strong>LLMs：</strong> LLMs通过分析和学习海量文本数据，掌握语言的结构、语义和上下文，具备广泛主题和任务的文本生成能力。然而，这些模型的准确性不仅依赖于丰富多样的训练数据和先进的模型架构，还面临着更新滞后和“幻觉”问题的挑战。在处理专业或最新信息时，LLMs可能产生无事实依据的内容，特别是在缺乏足够上下文或信息迅速变化的情况下。因此，在使用这些模型生成的信息进行决策和分析时，建议进行额外的验证，以提高准确性与可靠性。</p><p><strong>应用程序：</strong> 应用程序中的信息准确性高度依赖于内容的策划和管理，这通常涉及专业人员或团队的主动选择和提供。例如，在专业领域如医疗健康、金融投资等应用中，开发者和专家会投入大量资源来确保所提供信息的准确性、可靠性和及时更新。这种方法虽然可以提高特定领域内信息的质量，但同时也意味着用户接触到的信息范围受限于这些专业团队的知识广度和选择。因此，虽然应用程序在特定领域内可能提供高度准确的信息，它们的内容覆盖和视角受到了人为选择和专业领域限制的约束。</p><h4 id="可预期性" tabindex="-1"><a class="header-anchor" href="#可预期性"><span>可预期性</span></a></h4><p><strong>LLMs：</strong> LLMs的可预期性较低，因为它们生成的回答可能因训练数据的多样性和模型理解的复杂性而变化。用户可能会在没有明确指示的情况下收到多种可能的答案，或在特定情境下获得意料之外的回应，更差的情况是无法进行下一轮交互。这种不确定性源自于模型的设计和操作机制，尤其是在处理模糊查询或复杂问题时更为明显。</p><p><strong>应用程序：</strong> 应用程序通常提供更高的可预期性，因为它们是为了满足特定的需求和目的而设计，其功能、操作流程和用户界面都旨在提供一致和可预测的用户体验。应用程序通过明确的用户指令和固定的交互逻辑来减少操作过程中的不确定性，使用户能够预见到自己的操作将产生何种结果。</p><h4 id="可用性" tabindex="-1"><a class="header-anchor" href="#可用性"><span>可用性</span></a></h4><p><strong>LLMs：</strong> LLMs当前存在着大量可用性问题，除了上面提到的幻觉问题，还存在且不限于以下问题：</p><ol><li><strong>答非所问：</strong> LLMs在处理复杂或模糊的查询时，有时可能会提供与用户期望不符的答案。这种现象通常是由于模型对问题的理解不足或错误解释查询的意图所致。尽管模型训练涵盖了大量数据，以提高对自然语言的理解能力，但在理解特定上下文或领域特定语言时，仍可能遇到困难。</li><li><strong>响应时长：</strong> 用户在与LLMs进行交互时，尤其是在进行复杂的生成任务或处理大量数据请求时，可能会遇到较长的响应时间。这不仅影响用户体验，也限制了LLMs在实时交互场景中的应用。响应时间的延长主要由于模型的计算需求高和服务器负载大。</li><li><strong>处理能力的限制：</strong> 尽管LLMs的理论能力很强，但在实际应用中，其处理能力受到硬件资源、算法效率和可用性的制约。例如，大规模并发请求可能导致性能下降，特别是在资源受限的环境中。此外，LLMs在处理长篇文本或需要深入理解的复杂任务时，效率和准确性可能会降低。</li><li><strong>自我更新的局限：</strong> LLMs通常在固定的数据集上进行预训练，这意味着它们在训练之后的知识是静态的。尽管可以通过周期性的重新训练来更新模型，但是这种方法可能不足以实时捕捉最新信息或趋势，从而在一定程度上限制了模型在快速变化领域（例如新闻、科技等）的应用效果。</li></ol><p><strong>应用程序：</strong> 由于软件开发和设计领域长期以来的积累与创新，多数应用程序已经达到了高度的用户友好性和稳定性。设计师和开发者能够依托成熟的平台和框架来构建应用，同时，广泛采用的用户体验设计原则确保了应用能够满足用户的需求和期望。此外，通过持续的用户反馈和迭代开发，应用程序能够及时修复Bug，优化性能，从而提高用户满意度和应用的整体可用性。</p><h3 id="_1-3-开源llms有哪些优缺点" tabindex="-1"><a class="header-anchor" href="#_1-3-开源llms有哪些优缺点"><span>1.3 开源LLMs有哪些优缺点？</span></a></h3><h4 id="优点" tabindex="-1"><a class="header-anchor" href="#优点"><span>优点</span></a></h4><ol><li><strong>灵活性和适配性：</strong> 开源LLMs提供了不同尺寸的模型，使它们能够部署在各种设备上，从小型设备如树莓派和手机，到个人电脑乃至服务器集群。</li><li><strong>减少依赖：</strong> 使用开源LLMs可以在一定程度上减少对特定供应商，如OpenAI的依赖。这不仅提高了技术的可访问性，还促进了创新和自主性。</li><li><strong>隐私保护：</strong> 开源LLMs允许用户在本地或私有云环境中部署和使用模型，这有助于减少数据泄露的风险，为处理敏感或私有数据的应用提供了更高的安全性。</li><li><strong>可定制性：</strong> 开源模型可以根据特定的需求进行微调，使用户能够优化模型以适应特定的任务或数据集，从而提高了模型的效果和准确性。</li><li><strong>社区支持：</strong> 开源LLMs通常受益于活跃的开发者和研究社区的支持。这样的社区不仅提供技术支持和共享解决方案，还促进了新功能、优化和应用场景的快速创新。</li><li><strong>高透明度：</strong> 开源模型提供了算法和数据处理流程的可见性，使得技术用户和开发者能够审查模型的工作原理。</li></ol><h4 id="缺点" tabindex="-1"><a class="header-anchor" href="#缺点"><span>缺点</span></a></h4><ol><li><strong>基础模型和数据集的限制：</strong> 开源LLMs的性能上下限很大程度上取决于背后的基础模型和数据集。Meta、Mistral等厂商提供的模型虽然强大，但可能不适合所有类型的任务，特别是那些需要高度专业化知识的领域。</li><li><strong>资源和技术门槛：</strong> 尽管开源LLMs提供了极大的灵活性，但部署和维护这些模型需要相当的硬件资源和技术专长。对于缺乏这些资源或技术能力的个人或小团队来说，这可能是一个挑战。</li><li><strong>缺乏统一标准：</strong> 开源社区中存在多种LLMs，它们可能采用不同的架构、训练数据集和接口。这种多样性虽然促进了创新，但也给用户选择合适模型和工具带来了复杂性。</li><li><strong>稳定性问题：</strong> 本地部署的开源模型可能会遇到硬件限制和软件兼容性问题，导致运行不稳定。这种不稳定性可能会影响到模型的可靠性和用户体验。</li><li><strong>处理速度问题：</strong> 如果仅使用CPU进行模型推理，尤其是在处理大型模型或复杂任务时，速度可能非常慢。这不仅延长了处理时间，也可能限制模型在实时或要求高响应速度的应用场景中的使用。</li><li><strong>输出质量波动与内容生成的可控性：</strong> 开源LLMs在处理特定语言类型或复杂、边缘案例时，可能会遇到输出质量的波动，有时产生不相关或无意义的回答。同时，当需要生成敏感或要求高度准确的内容时，这些模型可能难以精确控制输出的质量和方向，特别是在缺乏细致的监督和定制化微调的情况下，这可能导致输出内容不达标或不满足特定的期望。</li></ol><h3 id="_1-4-线上和本地llms的区别" tabindex="-1"><a class="header-anchor" href="#_1-4-线上和本地llms的区别"><span>1.4 线上和本地LLMs的区别</span></a></h3><p><em>这里的LLMs特指Llama、Mistral、GLM等开源模型</em></p><h4 id="可用性-1" tabindex="-1"><a class="header-anchor" href="#可用性-1"><span>可用性</span></a></h4><p><strong>线上LLMs：</strong> 线上部署的LLMs提供即时访问和高可用性，这得益于SaaS（软件即服务）提供商在云服务器上预配置好的LLM和RAG（检索增强的生成模型）环境。用户无需担心硬件配置或安装过程，可以立即开始使用这些模型进行文本生成、问答等任务。这种部署方式特别适合没有专业技术背景的用户或需要快速部署解决方案的企业。</p><p><strong>本地LLMs：</strong> 本地部署的LLMs要求用户具备一定的技术知识，包括安装、配置和优化模型的能力。LLM的推理性能和速度直接受限于个人或组织的硬件配置，如处理器、内存和存储空间等。此外，虽然本地部署为用户提供了更大的控制空间，但缺少像RAG这样的高级功能封装，用户可能需要自己进行额外的开发工作。</p><h4 id="成本" tabindex="-1"><a class="header-anchor" href="#成本"><span>成本</span></a></h4><p><strong>线上LLMs：</strong> 对个人用户来说，线上LLMs服务的按需计费模式提供了极大的灵活性和入门门槛较低的优势。个人用户可以根据自己的实际需求和使用频率选择合适的服务计划，避免了高昂的初始投资。这对于那些只是偶尔需要使用语言模型进行特定项目或研究的用户尤其有利。然而，如果个人用户频繁使用这些服务进行大量数据处理，成本可能会逐渐累积，特别是当使用高级功能或大规模数据集时。对于那些需要长期、持续使用语言模型服务的个人用户，定期评估总成本是很重要的。</p><p><strong>本地LLMs：</strong> 对个人用户来说，选择本地部署LLMs意味着需要一次性投资于高性能的计算硬件。尽管这可能增加一些用户的经济成本，但它提供了长期的成本效益，尤其是对于那些有持续高强度使用需求的用户。本地部署避免了重复的服务费用，一旦设备投入使用，除了可能的维护和升级外，额外的运营成本相对较低。此外，个人用户通过本地部署能够获得更大的控制权和自定义能力，这可能对于研究人员或开发者特别有价值。然而，需要注意的是，本地部署也意味着用户必须具备一定的技术能力来配置和维护系统。</p><h4 id="隐私性" tabindex="-1"><a class="header-anchor" href="#隐私性"><span>隐私性</span></a></h4><p><strong>线上LLMs：</strong> 当使用线上LLMs时，用户的数据需要传输到云服务器上进行处理，这引发了对数据隐私和安全的考量。虽然许多SaaS提供商采取了强有力的数据保护措施，并承诺保护用户数据不被滥用或泄露，但这一过程仍然需要用户对提供商的数据处理和隐私政策有一定的信任基础。</p><p><strong>本地LLMs：</strong> 相对于线上模型，本地部署的LLMs在隐私保护方面提供了更高级别的安全性，主要因为数据处理在用户的私有设备或内部服务器上完成，无需数据外传。这种部署方式让用户对数据的控制权大大增强，降低了数据泄露的风险。</p><h4 id="依赖性和控制权" tabindex="-1"><a class="header-anchor" href="#依赖性和控制权"><span>依赖性和控制权</span></a></h4><p><strong>线上LLMs：</strong> 使用线上服务，用户依赖服务提供商确保服务的可用性和性能。这种模式简化了使用流程，允许用户专注于模型的应用而非其维护。然而，这也意味着在系统提示、上下文管理及模型响应定制方面，用户的直接控制能力有所限制。尽管线上服务提供一定程度的配置选项，但它们可能不足以满足所有特定需求，特别是在需要高度定制化输出的场景中。</p><p><strong>本地LLMs：</strong> 本地部署的模型让用户享有更高的控制权，包括对数据处理、模型配置和系统安全的管理。用户可以根据需要深度定制系统提示和上下文处理策略，这在特定应用场景下可能非常重要。然而，这种控制权和灵活性的增加伴随着更高的技术要求和可能的初期设置复杂性。尽管本地部署允许高度定制，但它也要求用户具备相应的技术能力来实现这些定制化的解决方案。</p><h4 id="透明度" tabindex="-1"><a class="header-anchor" href="#透明度"><span>透明度</span></a></h4><p><strong>线上LLMs：</strong> 线上LLMs服务由第三方提供，可能会在模型的工作原理和数据处理方式上给某些用户带来透明度的担忧。服务提供商通常会努力提供模型训练、数据处理和隐私政策等方面的文档，旨在提高透明度。然而，由于商业保密和操作复杂性，用户可能无法获得模型内部机制的完全细节。这要求用户信任服务提供商，并依赖其提供的信息和控制措施来保障数据安全和隐私。</p><p><strong>本地LLMs：</strong> 本地部署的LLMs允许用户直接访问模型，提供了更高程度的透明度。用户可以自行检查、修改和优化模型，从而深入理解其工作原理并根据需求调整其行为。这种直接控制确保了对模型的完全理解和定制能力，特别适合对数据安全、隐私保护有高要求或需遵循特定法规的组织。然而，这也意味着用户需要承担更大的责任，包括维护模型的透明度和确保其符合伦理和法律标准。</p><h4 id="离线" tabindex="-1"><a class="header-anchor" href="#离线"><span>离线</span></a></h4><p><strong>线上LLMs：</strong> 无法使用</p><p><strong>本地LLMs：</strong> 可正常使用</p><h3 id="_1-5-使用线上llms存在哪些问题" tabindex="-1"><a class="header-anchor" href="#_1-5-使用线上llms存在哪些问题"><span>1.5 使用线上LLMs存在哪些问题？</span></a></h3><p>在使用线上LLMs时，我们面临着多个层面的挑战，这些问题从个人数据隐私到模型性能与成本，再到内容的完整性以及检索增强的生成模型（RAG）策略的选择和应用，构成了一系列需要用户和服务提供商共同关注的关键问题。</p><h4 id="问题1-个人数据隐私" tabindex="-1"><a class="header-anchor" href="#问题1-个人数据隐私"><span>问题1：个人数据隐私</span></a></h4><p>在线上LLMs的使用过程中，个人数据的隐私成为了一个显著的担忧点。用户与LLM的互动，以及上传的文档，都可能被服务提供商收集用于模型的微调和优化。</p><h4 id="问题2-模型性能与成本" tabindex="-1"><a class="header-anchor" href="#问题2-模型性能与成本"><span>问题2：模型性能与成本</span></a></h4><p>服务提供商为了平衡成本和性能，可能会使用不同规模的LLM。较小的模型（如7B或13B参数）虽然可以减少计算资源消耗，提高响应速度，但其推理能力可能不足以处理复杂的查询或生成高质量的文本。用户在选择服务时，可能难以获悉所使用的模型规模和性能，导致实际应用中的效果与预期存在差异。这种不透明性可能会影响用户的使用体验和满意度。</p><h4 id="问题3-内容完整性" tabindex="-1"><a class="header-anchor" href="#问题3-内容完整性"><span>问题3：内容完整性</span></a></h4><p>针对长文档的处理，存在一个问题是LLM是否能够完整地获取和理解整个文档内容。由于技术限制，一些服务可能只会处理文档的一部分内容，如仅分析前几百个词。这种处理方式可能会导致LLM遗漏关键信息，从而影响到生成内容的准确性和相关性。用户可能无法了解其提交的内容是否被完整处理，进而对结果的可靠性产生疑问。</p><h4 id="问题4-检索增强的生成模型-rag-策略" tabindex="-1"><a class="header-anchor" href="#问题4-检索增强的生成模型-rag-策略"><span>问题4：检索增强的生成模型（RAG）策略</span></a></h4><p>RAG的切割方式和回调策略直接影响LLM的效果，特别是在处理需要广泛知识检索的查询时。不同的切割方法和回调策略决定了LLM访问和整合信息的方式，从而影响到最终生成的答案的准确性和完整性。如果这些策略没有恰当选择或优化，LLM可能无法看到或利用相关的信息来生成最佳的回答。用户通常无法控制或知晓这些内部处理细节，这增加了结果不确定性。</p><h2 id="_2-哪些理论场景适合使用本地开源llms" tabindex="-1"><a class="header-anchor" href="#_2-哪些理论场景适合使用本地开源llms"><span>2. 哪些理论场景适合使用本地开源LLMs？</span></a></h2><p><em>此处标注理论场景是因为当前开源LLMs存在较多实际问题，但这些问题将被逐步克服。</em></p><h4 id="_1-我的业务场景是否存在大容量、高频的数据请求" tabindex="-1"><a class="header-anchor" href="#_1-我的业务场景是否存在大容量、高频的数据请求"><span>1. 我的业务场景是否存在大容量、高频的数据请求？</span></a></h4><p>如果你每天需都需要分析和处理大量文档，而一篇文档大概需要4000Token，LLM一轮输出400 Token，那么10轮交互下来最少需要5W Token，20篇那就消耗了100W Token；如果需要跟之前的文档进行交叉比较和深度分析，假设库里有1000篇文章，那一天需要消耗10亿Token（前提是没有其他技术的加持）。</p><p><strong>理论场景：</strong> 个人知识库场景、多文档自动分析场景......</p><h4 id="_2-我的业务场景是否涉及多重信息加工甚至多任务" tabindex="-1"><a class="header-anchor" href="#_2-我的业务场景是否涉及多重信息加工甚至多任务"><span>2. 我的业务场景是否涉及多重信息加工甚至多任务？</span></a></h4><p>在多任务和多重信息加工领域，涉及到的操作和处理逻辑非常复杂，尤其是在电脑上进行的活动。这些活动不仅仅是简单的执行命令，更多的是涉及到对用户意图的判断、数据之间的传输和交互，以及如何根据当前的上下文来做出响应。这里面涉及到的Token消耗主要是因为这些操作需要大量的计算资源来处理和响应用户的需求。</p><p>在这个过程中，每一个步骤所需的系统提示（System Prompt）、上下文对话的逻辑处理等，都需要根据具体情况进行个性化和差异化的设计。这种情况下的“千人千面”，是指系统必须能够根据每个用户的具体需求和上下文来提供定制化的服务。这与基于推荐算法的“千人千面”有着本质的差异。本质上的差异在于：</p><ol><li><strong>处理的动态性与静态性：</strong> 多任务和多重信息加工领域的处理是动态的，依赖于当前的上下文和用户的即时需求，而基于推荐算法的处理更多是静态的，依赖于用户的历史数据。</li><li><strong>个性化程度：</strong> 虽然两者都追求“千人千面”，但多任务处理更加注重在特定情境下对用户需求的精准响应，推荐算法更多是依据历史数据或者类似用户的数据进行的一般性预测。</li><li><strong>交互性：</strong> 多任务处理涉及到与用户的实时交互，而推荐算法则更多是一种单向的内容推送。</li></ol><p><em>请注意，这个本质的差异决定了本地LLMs是解决多重信息加工甚至多任务的唯一解决办法，但它有可能是一个闭源LLM。</em></p><p><strong>理论场景：</strong> 撰写报告场景、依赖操作系统能力的交互场景......</p><h4 id="_3-我的业务场景是否涉及到个人隐私或敏感数据" tabindex="-1"><a class="header-anchor" href="#_3-我的业务场景是否涉及到个人隐私或敏感数据"><span>3. 我的业务场景是否涉及到个人隐私或敏感数据？</span></a></h4><p>在探讨目标设定、日记撰写、财务管理、情感交流等个人化领域时，我们不得不面对一个现实问题：这些活动往往涉及到大量的个人数据处理，而这些数据中包含了用户的敏感信息。比如，在进行日记撰写时，用户可能会记录下自己的私密想法和感受；在财务管理过程中，用户需要处理自己的收入、支出等财务信息；情感交流时，用户可能会分享自己的心理状态和情绪体验。这些活动中生成和处理的数据，都属于高度敏感的个人信息，一旦被泄露，可能会给用户造成不可挽回的伤害。在这样的背景下，本地开源LLMs的应用就显得尤为重要和必要。</p><p><strong>理论场景：</strong> 个人目标设定与跟踪场景、私密情感交流场景、日记撰写与情感分析场景......</p><h4 id="_4-我的业务场景是否需要低护栏的大模型" tabindex="-1"><a class="header-anchor" href="#_4-我的业务场景是否需要低护栏的大模型"><span>4. 我的业务场景是否需要低护栏的大模型？</span></a></h4><p>在当前的技术环境中，线上LLM在提供服务的同时，不得不面对各种法律法规和伦理道德的约束。这些约束往往通过增设“护栏”来实施，目的是为了防止模型生成不当内容，如误导信息、侵犯隐私、散播仇恨言论等。然而，这些护栏在一定程度上限制了LLM的功能和知识的完整使用，尤其是在需要较高自由度的应用场景中，如角色扮演、创意写作等。在这些场景下，内容的过滤和限制可能会阻碍创意的发挥和用户体验的深化。</p><p>在此背景下，本地部署的LLMs提供了一种解决方案。通过本地部署，用户可以根据自己的需求和责任范围，调整内容过滤的标准，既能规避法律风险，又能保留更高的创意自由度和个性化服务。这种灵活性在某些业务场景中尤为重要，能够为用户提供更加丰富和深入的互动体验。</p><p><strong>理论场景：</strong> 部分领域的咨询场景、角色扮演场景</p><h2 id="_3-开箱即用的本地llms" tabindex="-1"><a class="header-anchor" href="#_3-开箱即用的本地llms"><span>3. 开箱即用的本地LLMs</span></a></h2><p>得益于Meta、Mistral和众多开发者提供的开源LLMs，以及Llama.cpp这个开源项目，目前我们可以以很低的成本使用开源LLMs，接下来我推荐一些开箱即用的后端和客户端来帮助各位开启本地LLMs之旅。</p><h3 id="本地llms推荐及下载" tabindex="-1"><a class="header-anchor" href="#本地llms推荐及下载"><span>本地LLMs推荐及下载</span></a></h3><p><em>我们推荐使用GGUF版本的量化模型，因为它可以在CPU和消费级GPU环境下单独或者混合使用。欢迎分享自己使用过不错的模型给我们</em></p><h4 id="_1-qwen1-5-32b" tabindex="-1"><a class="header-anchor" href="#_1-qwen1-5-32b"><span>1. Qwen1.5-32B</span></a></h4><p><strong>介绍：</strong> Qwen1.5-32B是阿里云研发的通义千问大模型系列的320亿参数规模的模型。我们推荐使用Q2及以上量化模型。</p><p><strong>官网：</strong> https://github.com/QwenLM/Qwen1.5</p><p><strong>模型大小：</strong> 32B</p><p><strong>相关语言：</strong> 英文、中文</p><p><strong>上下文长度：</strong> 32k</p><p><strong>适用于：</strong> 拥有24GB显存及以上的英伟达显卡电脑、拥有32GB内存及以上的M1/M2/M3 Mac</p><p><strong>下载地址：</strong></p><p>https://huggingface.co/Qwen/Qwen1.5-32B-Chat-GGUF</p><p>https://modelscope.cn/models/qwen/Qwen1.5-32B-Chat-GGUF/summary</p><h4 id="_2-qwen1-5-14b" tabindex="-1"><a class="header-anchor" href="#_2-qwen1-5-14b"><span>2. Qwen1.5-14B</span></a></h4><p><strong>介绍：</strong> Qwen1.5-14B是阿里云研发的通义千问大模型系列的140亿参数规模的模型。我们推荐使用Q2及以上量化模型。</p><p><strong>官网：</strong> https://github.com/QwenLM/Qwen1.5</p><p><strong>模型大小：</strong> 14B</p><p><strong>相关语言：</strong> 英文、中文</p><p><strong>上下文长度：</strong> 32k</p><p><strong>适用于：</strong> 拥有12GB显存及以上的英伟达显卡电脑、拥有16GB内存及以上的M1/M2/M3 Mac</p><p><strong>下载地址：</strong></p><p>https://huggingface.co/Qwen/Qwen1.5-14B-Chat-GGUF</p><p>https://modelscope.cn/models/qwen/Qwen1.5-14B-Chat-GGUF/summary</p><h4 id="_3-qwen1-5-7b" tabindex="-1"><a class="header-anchor" href="#_3-qwen1-5-7b"><span>3. Qwen1.5-7B</span></a></h4><p><strong>介绍：</strong> Qwen1.5-7B是阿里云研发的通义千问大模型系列的70亿参数规模的模型。我们推荐使用Q2及以上量化模型。</p><p><strong>官网：</strong> https://github.com/QwenLM/Qwen1.5</p><p><strong>模型大小：</strong> 7B</p><p><strong>相关语言：</strong> 英文、中文</p><p><strong>上下文长度：</strong> 32K</p><p><strong>适用于：</strong> 适用于大部分电脑，但建议拥有12GB显存及以上的英伟达显卡电脑、拥有16GB内存及以上的M1/M2/M3 Mac</p><p><strong>下载地址：</strong></p><p>https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GGUF</p><p>https://modelscope.cn/models/qwen/Qwen1.5-7B-Chat-GGUF/summary</p><h4 id="_4-mixtral-8x7b-instruct-v0-1" tabindex="-1"><a class="header-anchor" href="#_4-mixtral-8x7b-instruct-v0-1"><span>4. Mixtral-8x7B-Instruct-v0.1</span></a></h4><p><strong>介绍：</strong> Mixtral 8x7B是一个具有开放权重的专家模型 （MoE） 混合。我们推荐使用Q3及以上的量化模型。</p><p><strong>官网：</strong> https://mistral.ai/</p><p><strong>模型大小：</strong> 8x7B</p><p><strong>相关语言：</strong> 多语言</p><p><strong>上下文长度：</strong> 32K</p><p><strong>适用于：</strong> 拥有24GB显存及以上的英伟达显卡电脑、拥有32GB内存及以上的M1/M2/M3 Mac</p><p><strong>下载地址：</strong></p><p>https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/tree/main</p><p>https://modelscope.cn/models/limoncc/Mixtral-8x7B-Instruct-v0.1-GGUF/files</p><h4 id="_5-openchat-3-5-0106" tabindex="-1"><a class="header-anchor" href="#_5-openchat-3-5-0106"><span>5. OpenChat 3.5 0106</span></a></h4><p><strong>介绍：</strong> OpenChat在24年1月份的版本，通过 C-RLFT 在Mistral 7B的基础上进行微调，在推理和中英文双语问答上有较好的表现。我们推荐使用Q4及以上量化模型。</p><p><strong>官网：</strong> https://github.com/imoneoi/openchat</p><p><strong>模型大小：</strong> 7B</p><p><strong>相关语言：</strong> 中文、英文</p><p><strong>上下文长度：</strong> 8K</p><p><strong>适用于：</strong> 适用于大部分电脑，但建议拥有12GB显存及以上的英伟达显卡电脑、拥有16GB内存及以上的M1/M2/M3 Mac</p><p><strong>下载地址：</strong></p><p>https://huggingface.co/TheBloke/openchat-3.5-0106-GGUF</p><p>https://modelscope.cn/models/fivetwin/openchat-3.5-0106-GGUF/files</p><h4 id="_6-openchat-3-5-16k" tabindex="-1"><a class="header-anchor" href="#_6-openchat-3-5-16k"><span>6. OpenChat 3.5-16k</span></a></h4><p><strong>介绍：</strong> OpenChat在23年12月份的版本，通过 C-RLFT 在Mistral 7B的基础上进行微调，在推理和中英文双语问答上有较好的表现。我们推荐使用Q4及以上量化模型。</p><p><strong>官网：</strong> https://github.com/imoneoi/openchat</p><p><strong>模型大小：</strong> 7B</p><p><strong>相关语言：</strong> 中文、英文</p><p><strong>上下文长度：</strong> 16K</p><p><strong>适用于：</strong> 适用于大部分电脑，但建议拥有12GB显存及以上的英伟达显卡电脑、拥有16GB内存及以上的M1/M2/M3 Mac</p><p><strong>下载地址：</strong></p><p>https://huggingface.co/TheBloke/openchat_3.5-16k-GGUF/tree/main</p><p>https://modelscope.cn/models/limoncc/OPENCHAT3.5-16K-GGUF/files</p><h4 id="_7-llava-1-6" tabindex="-1"><a class="header-anchor" href="#_7-llava-1-6"><span>7. llava 1.6</span></a></h4><p><strong>介绍：</strong> LLaVA 是用于聊天机器人的多模态模型，拥有计算机视觉和自然处理能力，可以分析图片并用文字进行反馈。</p><p><strong>模型大小：</strong> 7B</p><p><strong>相关语言：</strong> 英文</p><p><strong>上下文长度：</strong> 32K</p><p><strong>适用于：</strong> 适用于大部分电脑，但建议拥有12GB显存及以上的英伟达显卡电脑、拥有16GB内存及以上的M1/M2/M3 Mac</p><p><strong>注意事项：</strong> 下载文件时除了量化模型，还要下载配置文件以及对应的mmproj模型，这些文件夹需要放在同一个文件夹内。</p><p><strong>下载地址：</strong></p><p>https://huggingface.co/cmp-nct/llava-1.6-gguf/</p><p>https://modelscope.cn/models/mirror013/llava-1.6-mistral-7b-gguf/summary</p><h4 id="_8-gemma-7b-it" tabindex="-1"><a class="header-anchor" href="#_8-gemma-7b-it"><span>8. Gemma-7b-it</span></a></h4><p><strong>介绍：</strong> Gemma 是 Google 推出的一系列轻量级、最先进的开放模型，采用与创建 Gemini 模型相同的研究和技术构建。它们是文本到文本、仅限解码器的大型语言模型，提供英语版本，具有开放权重、预训练变体和指令调整变体。</p><p><strong>官网：</strong> https://huggingface.co/google/gemma-7b-it</p><p><strong>模型大小：</strong> 7B</p><p><strong>相关语言：</strong> 多语言</p><p><strong>上下文长度：</strong> 8K</p><p><strong>适用于：</strong> 适用于大部分电脑，但建议拥有12GB显存及以上的英伟达显卡电脑、拥有16GB内存及以上的M1/M2/M3 Mac</p><p><strong>注意事项：</strong> 在llama.cpp上使用Q8以下的量化模型会有Bug。</p><p><strong>下载地址：</strong></p><p>https://huggingface.co/sayhan/gemma-7b-it-GGUF-quantized</p><p>https://modelscope.cn/models/fivetwin/gemma-7b-it-gguf/summary</p><h4 id="_9-gemma-2b-it" tabindex="-1"><a class="header-anchor" href="#_9-gemma-2b-it"><span>9. Gemma-2b-it</span></a></h4><p><strong>介绍：</strong> Gemma 是 Google 推出的一系列轻量级、最先进的开放模型，采用与创建 Gemini 模型相同的研究和技术构建。它们是文本到文本、仅限解码器的大型语言模型，提供英语版本，具有开放权重、预训练变体和指令调整变体。</p><p><strong>官网：</strong> https://huggingface.co/google/gemma-2b-it</p><p><strong>模型大小：</strong> 2B</p><p><strong>相关语言：</strong> 多语言</p><p><strong>上下文长度：</strong> 8K</p><p><strong>适用于：</strong> 适用于大部分电脑</p><p><strong>注意事项：</strong> 在llama.cpp上使用Q8以下的量化模型会有Bug。</p><p><strong>下载地址：</strong></p><p>https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF</p><p>https://modelscope.cn/models/CruiseTian/gemma-2b-gguf-quantized/summary</p><h3 id="如何开启本地llm" tabindex="-1"><a class="header-anchor" href="#如何开启本地llm"><span>如何开启本地LLM</span></a></h3><h4 id="_1-lm-studio" tabindex="-1"><a class="header-anchor" href="#_1-lm-studio"><span>1. LM Studio</span></a></h4><p><img src="'+i+'" alt="1.png"></p><p><strong>介绍</strong>：LM Studio基于LLama.cpp开发了一套友好的用户界面，支持在Windows、Mac和Linux系统上安装和使用，提供了多种语言模型，并且提供了模型搜索、下载和聊天等功能。</p><p><strong>官网及下载地址</strong>：https://lmstudio.ai/</p><p><strong>操作系统：</strong> Windows、Mac、Linux</p><p><strong>相关语言：</strong> 英文</p><p><strong>使用步骤：</strong></p><ol><li>下载并安装软件。</li><li>在应用内下载模型，如果因网络等情况无法下载，可以选择存放LLM的文件夹后并在文件夹里新建一个名为“TheBloke”的文件夹，然后将下载过的模型放进“TheBloke”文件夹里。 <img src="'+g+'" alt="2.png"></li><li>点击”Select a model to load“选择模型即可开启对话模式，也可以根据自己电脑和模型选择对应的“GPU Acceleration”和“Context Length”，重新加载模型。（如果自己的电脑配置一般，建议不勾选”GPU Offload“，以及“Context Length”设置越大，等待时间会越长，更详细的介绍请看指南第四部分） <img src="'+h+'" alt="3.png"></li><li>如果开启服务端模式，加载模型并填写端口（例如8000），点击“Start Server”即可运行。记住，客户端和服务端的通讯必须兼容OpenAI API模式。 <img src="'+c+'" alt="4.png"></li></ol><p><strong>优点：</strong></p><ol><li>整体使用流程对普通用户友好。</li><li>可以很方便地切换模型。</li><li>可以很方便地选择CPU和GPU的混合程度。</li><li>可以很方便地切换Preset。</li><li>兼容多模态模型，可以在界面里直接查询图片。</li></ol><p><strong>注意事项：</strong></p><ol><li>切换模型或者设置参数后一定要记得重新手动加载模型。</li><li>服务端模式下，重新更换模型或者设置参数后，有可能需要重启服务端。</li><li>暂时不支持开机启动。</li><li>暂时不支持自动进入服务端模式。</li><li>如果你的系统是Windows/Linux，处理器需要支持AVX2。</li></ol><h4 id="_2-llamafile" tabindex="-1"><a class="header-anchor" href="#_2-llamafile"><span>2. llamafile</span></a></h4><p><img src="'+L+'" alt="5.png"></p><p><strong>介绍</strong>：llamafile将 llama.cpp 与 Cosmopolitan Libc 组合成一个框架，将 LLMs 的所有复杂性压缩为在大多数计算机上本地运行的单个文件可执行文件。</p><p><strong>官网：</strong> https://github.com/Mozilla-Ocho/llamafile</p><p><strong>下载地址：</strong> https://github.com/Mozilla-Ocho/llamafile/releases/</p><p><strong>操作系统：</strong> Windows、Mac、Linux</p><p><strong>相关语言：</strong> 英文</p><p><strong>使用步骤：</strong></p><ol><li>下载文件</li><li>如果你是macOS或者Linux用户，打开计算机终端，授予计算机执行此新文件的权限<code>chmod +x llava-v1.5-7b-q4.llamafile</code>，接着运行该llama文件<code>./llava-v1.5-7b-q4.llamafile -ngl 9999</code>。（记住先进入对应的文件夹再输入指令，“llava-v1.5-7b-q4.llamafile”是你下载的文件名字）</li><li>如果你是Windows用户，可以在文件末尾添加“.exe”来重命名该文件。</li><li>你的浏览器应该自动打开并显示聊天界面。 （如果没有，只需打开浏览器并将其指向 http://127.0.0.1:8080/</li><li>聊天完毕后，返回终端并点击 <code>Control-C</code> 关闭 llamafile，或者直接关闭cmd窗口。</li><li>另外，开启llamafile后会自动开启服务端模式，在客户端填写 http://127.0.0.1:8080/ 即可通讯。记住，客户端和服务端的通讯必须兼容OpenAI API模式。</li></ol><p><strong>优点：</strong></p><ol><li>下载文件后即可直接使用。</li><li>开启应用后直接进入服务端模式。</li><li>可以通过命令行方式设置的方式开机启动应用和服务端模式。</li><li>兼容多模态模型，可以在界面里直接查询图片。</li></ol><p><strong>注意事项：</strong></p><ol><li>目前llamafile的一部分功能依赖于CLI（Command Line Interface），更适合拥有编程能力的用户使用。</li><li>由于Windows对可执行文件大小有4GB的限制，所以大部分从网上下载的llamafiles无法使用，需要自己重新打包和设置llamafile，详情请看： https://github.com/Mozilla-Ocho/llamafile 。</li><li>导入本地已经下载好的模型，需要重新打包另外一个llamafile。</li><li>使用过程中更换模型约等于打开另外一个文件，操作起来不怎么方便。</li><li>如果你的系统是Windows/Linux，处理器需要支持AVX2。</li></ol><h4 id="_3-gpt4all" tabindex="-1"><a class="header-anchor" href="#_3-gpt4all"><span>3. GPT4All</span></a></h4><p><img src="'+d+'" alt="20.png"><strong>介绍</strong>：一个免费使用、本地运行、具有隐私意识的聊天机器人，无需 GPU 或互联网。</p><p><strong>官网及下载地址</strong>：https://gpt4all.io/</p><p><strong>操作系统：</strong> Windows、Mac、Linux</p><p><strong>相关语言：</strong> 英文</p><p><strong>使用步骤：</strong></p><ol><li>下载并安装软件。</li><li>选择模型存放路径。</li><li>下载LLMs和RAG模型（如果模型存放路径里有模型，可以直接使用）。</li><li>在应用顶部选择模型后即可对话。</li><li>如果需要开启Server模式，点击一下Wifi图标即可，但无法查看和修改端口。Server模式下端口是4891，完整URL：http://127.0.0.1:4891。</li></ol><p><strong>优点：</strong></p><ol><li>支持RAG功能。</li><li>可以选择CPU和GPU的混合程度。</li></ol><p><strong>注意事项：</strong></p><ol><li>在应用内无法查看和修改Server模式下的端口（4891）。</li><li>GPT4All有可能没有兼容完整的OpenAI API模式，部分客户端接收不到数据。</li><li>暂时不支持开机启动。</li><li>暂时不支持自动进入服务端模式。</li><li>暂时不支持自动选择RAG的本地文件夹。</li><li>如果你的系统是Windows/Linux，处理器需要支持AVX2。</li><li>用户体验上有待提高。</li></ol><h4 id="_4-ollama" tabindex="-1"><a class="header-anchor" href="#_4-ollama"><span>4. Ollama</span></a></h4><p><strong>介绍</strong>：Ollama 是一个基于LLama.cpp的轻量级、可扩展的框架，用于在本地构建和运行语言模型。</p><p><strong>官网及下载地址</strong>：https://ollama.com/</p><p><strong>操作系统：</strong> Windows、Mac、Linux</p><p><strong>相关语言：</strong> 英文</p><p><strong>使用步骤：</strong></p><ol><li>下载并打开执行文件。</li><li>成功安装后会提示在cmd内运行模型，例如<code>ollama run mistral</code>（如果没有下载过模型会自动下载模型）。 <img src="'+m+'" alt="6.png"><img src="'+M+'" alt="7.png"></li><li>在cmd中开始对话</li></ol><p><strong>优点：</strong></p><ol><li>支持开机启动。</li><li>开启应用后直接进入服务端模式。</li><li>可以通过命令行直接下载模型。</li><li>可以通过命令行直接切换模型。</li></ol><p><strong>注意事项：</strong></p><ol><li>目前Ollama依赖于CLI（Command Line Interface），更适合拥有编程能力的用户使用。</li><li>导入本地已经下载好的模型需要命令行导入，详情请看： https://github.com/ollama/ollama 。</li><li>服务端模式跟OpenAI API不一样，需要客户端兼容并选择对应的模型。</li><li>如果你的系统是Windows/Linux，处理器需要支持AVX2。</li></ol><p><em>欢迎分享自己使用过不错的LLM后端给我们</em></p><h3 id="可连接本地llm的应用推荐" tabindex="-1"><a class="header-anchor" href="#可连接本地llm的应用推荐"><span>可连接本地LLM的应用推荐</span></a></h3><h4 id="_1-mix-copilot" tabindex="-1"><a class="header-anchor" href="#_1-mix-copilot"><span>1. MiX Copilot</span></a></h4><p><img src="'+_+'" alt="8.png"></p><p><strong>介绍</strong>：MiX Copilot 是一款支持OpenAI和本地LLMs的PC客户端，用于自动爬取、整理和分析资料（数据都以Markdown的形式保存到本地，避免隐私泄露），支持多个Chatbot同时对话、浏览网页+使用LLMs、制作LLM工作流（包括定制Prompt、查询网页）等功能。</p><p><strong>官网及下载地址</strong>：https://www.mix-copilot.com/</p><p><strong>操作系统：</strong> Windows、Mac</p><p><strong>相关语言：</strong> 中文、英文</p><p><strong>使用教程：</strong> https://hci-top.notion.site/MiX-Copilot-Tutorial-English-2c95481c5d4c4c818f40bd04c299b7ea</p><p><strong>使用步骤：</strong></p><ol><li>下载并安装软件</li><li>进入“设置-LLM设置”，在本地模型设置项填写服务端链接（记住端口号要一致），例如 http://127.0.0.1:8000 ，点击顶部的”更新“按钮，如果红色Tag变成绿色，说明和LLM服务端连接成功。 <img src="'+f+'" alt="9.png"></li><li>在顶部Tab点击“+”号，或者在浏览网页时右键点击“展示Chatbot”开启Chatbot。 <img src="'+b+'" alt="10.png"></li><li>在对话面板左下角选择本地LLM! <img src="'+x+'" alt="11.png"></li></ol><h4 id="_2-anythingllm-for-desktop" tabindex="-1"><a class="header-anchor" href="#_2-anythingllm-for-desktop"><span>2. AnythingLLM for Desktop</span></a></h4><p><img src="'+u+'" alt="15.png"><strong>介绍</strong>：AnythingLLM 可用于无限文档，完全私密，可以使用GPT-4，自定义模型或开源模型如 Llama、Mistral 等，支持 PDF、Word 文档等多种格式。</p><p><strong>官网及下载地址</strong>：https://useanything.com/</p><p><strong>操作系统：</strong> Windows、Mac</p><p><strong>相关语言：</strong> 英文</p><p><strong>使用步骤：</strong></p><ol><li>下载并安装软件。</li><li>分别设置模型（可以设置OpenAI API、连接LM-Studio等）、Embedding、向量数据库 <img src="'+G+'" alt="117.png"><img src="'+B+'" alt="18.png"><img src="'+w+'" alt="19.png"></li><li>设置工作区。</li><li>开始对话。</li><li>可以在客户端内添加MD、PDF、MP4等文件，或者直接添加网站URL。</li><li>将以上文档添加到工作区开始Embedding并进行对话。 <img src="'+T+'" alt="16.png"></li></ol><h4 id="_3-reor" tabindex="-1"><a class="header-anchor" href="#_3-reor"><span>3. Reor</span></a></h4><p><img src="'+k+'" alt="13.jpg"><strong>介绍</strong>：Reor 是一款由人工智能驱动的桌面笔记应用程序：它会自动链接相关想法、回答笔记中的问题并提供语义搜索。所有内容都存储在本地，用户可以使用类似黑曜石的 Markdown 编辑器来编辑笔记。</p><p><strong>官网及下载地址</strong>：https://www.reorproject.org/</p><p><strong>操作系统：</strong> Windows、Mac</p><p><strong>相关语言：</strong> 英文</p><p><strong>使用步骤：</strong></p><ol><li>下载并安装软件。</li><li>选择你存放Markdown的文件夹。</li><li>在设置页选择你下载好的量化模型（GGUF格式）以及设置上下文长度。 <img src="'+P+'" alt="14.png"></li><li>选择你需要的Embedding模型。<strong>注意：</strong> 一部分地区用户可能因为网络问题无法下载Embedding模型，导致应用无法使用。另外，应用中提供的Embedding模型只擅长处理英文，其他语言不一定适用。</li><li>设置RAG的返回数量。</li></ol><h4 id="_4-chat-with-rtx" tabindex="-1"><a class="header-anchor" href="#_4-chat-with-rtx"><span>4. Chat with RTX</span></a></h4><p><img src="'+A+'" alt="img.png"><strong>介绍</strong>：Chat With RTX 是一款<strong>演示</strong>应用程序，可以将自己的内容（文档、笔记、视频或其他数据）与本地LLM进行交互。Chat With RTX集成了检索增强生成 (RAG)、TensorRT-LLM 和 RTX 加速，用户可以查询自定义聊天机器人以快速获得上下文相关的答案。Chat with RTX 支持各种文件格式，包括文本、pdf、doc/docx 和 xml。只需将应用程序指向包含文件的文件夹，它就会在几秒钟内将它们加载到库中。此外，用户可以提供 YouTube 播放列表的网址，应用程序将加载播放列表中视频的转录，然后查询它们涵盖的内容。</p><p><strong>官网及下载地址</strong>：https://www.nvidia.com/en-us/ai-on-rtx/chat-with-rtx-generative-ai/</p><p><strong>操作系统：</strong> Windows</p><p><strong>相关语言：</strong> 英文</p><p><strong>注意事项：</strong></p><ol><li>目前Chat with RTX是一个演示应用程序。</li><li>目前Chat with RTX要求NVIDIA GeForce™ RTX 30 或 40 系列 GPU 或 NVIDIA RTX™ Ampere 或 Ada Generation GPU，具有至少 8GB VRAM 和 16GB RAM。</li><li>演示应用程序安装包体积高达35GB。</li></ol><h4 id="_5-待补充" tabindex="-1"><a class="header-anchor" href="#_5-待补充"><span>5. 待补充</span></a></h4><p><em>目前全球范围内可以兼容本地LLM的客户端极少，欢迎各位推荐或自荐</em></p><h2 id="_4-使用本地llms需要注意什么" tabindex="-1"><a class="header-anchor" href="#_4-使用本地llms需要注意什么"><span>4. 使用本地LLMs需要注意什么？</span></a></h2><h4 id="_1-首先你需要知道以下几点" tabindex="-1"><a class="header-anchor" href="#_1-首先你需要知道以下几点"><span>1. 首先你需要知道以下几点：</span></a></h4><ol><li>不同训练规模的LLM都会出错，即使是GPT-5。</li><li>不同训练规模的LLM决定了它的自身推理水平，例如13B优于2B的LLM模型，但是可以通过微调来提升小模型完成任务的质量。</li><li>LLM模型被量化后，体积越小，精度也会越差，例如Q2 &lt; Q8。</li><li>目前运行LLM的最佳设备是英伟达GPU或者M1及以上的Mac，如果你的电脑是AMD或者Intel的GPU，目前只能用CPU来运行LLM。</li><li>LLM模型全部加载到GPU并运行是最快的。</li></ol><h4 id="_2-如何让llm高效运行" tabindex="-1"><a class="header-anchor" href="#_2-如何让llm高效运行"><span>2. 如何让LLM高效运行？</span></a></h4><ol><li><strong>8G及以下的GPU</strong>：2B-LLM + 4-8k Token或者7B-LLM + 2-4k Token。</li><li><strong>8-12GB的GPU</strong>：7B-LLM + 4k-6k Token或者14B-LLM + 2-4k Token。</li><li><strong>12-24GB的GPU-</strong>：7B-LLM + 8-12k Token或者14B-LLM + 4-8k Token。</li></ol><p>注意：最终LLM模型体积 + 上下文占用的显存 &lt; GPU显存，否则速度会变慢（例如8k上下文会占用到6~10G的显存），你可以根据自己需求适当选择不同量化模型来减少LLM模型的体积。</p><h4 id="_3-模型中的q、s、k和数组分别代表什么" tabindex="-1"><a class="header-anchor" href="#_3-模型中的q、s、k和数组分别代表什么"><span>3. 模型中的Q、S、K和数组分别代表什么？</span></a></h4><h4 id="_4-当前llm经常出现的问题有哪些" tabindex="-1"><a class="header-anchor" href="#_4-当前llm经常出现的问题有哪些"><span>4. 当前LLM经常出现的问题有哪些？</span></a></h4><h4 id="_5-目前llm能做什么" tabindex="-1"><a class="header-anchor" href="#_5-目前llm能做什么"><span>5. 目前LLM能做什么？</span></a></h4><h4 id="_6-如何本地使用rag" tabindex="-1"><a class="header-anchor" href="#_6-如何本地使用rag"><span>6. 如何本地使用RAG？</span></a></h4><h4 id="_7-目前本地rag是否可用" tabindex="-1"><a class="header-anchor" href="#_7-目前本地rag是否可用"><span>7. 目前本地RAG是否可用？</span></a></h4><h2 id="_5-常见专业术语" tabindex="-1"><a class="header-anchor" href="#_5-常见专业术语"><span>5. 常见专业术语</span></a></h2><h4 id="system-prompt" tabindex="-1"><a class="header-anchor" href="#system-prompt"><span>System Prompt</span></a></h4><p><strong>解释：</strong> 在大型语言模型（LLM）的上下文中，系统提示是指模型给出的初始文本或指令，用于引导或初始化模型的行为。这个提示通常包含了让模型执行特定任务所需的信息或背景。例如，在一个聊天机器人应用中，系统提示可能是一系列固定的问题或指令，用于激活模型开始一个对话流程。</p><p><strong>注意事项：</strong> 虽然最终用户通常不直接设置系统提示，但了解它的概念有助于理解与语言模型交互的起点。对于开发者而言，在设计交互式应用或任务时精心设计系统提示，可以大幅提高模型执行任务的效率和准确性。</p><h4 id="user-prompt" tabindex="-1"><a class="header-anchor" href="#user-prompt"><span>User Prompt</span></a></h4><p><strong>解释：</strong> User Prompt是用户输入的文本，旨在引导或请求语言模型进行特定的响应或操作。比如，用户可能会回答系统的问题或直接提出自己的问题。</p><p><strong>注意事项：</strong> 用户在提供提示时应尽量明确具体，以便语言模型能更准确地理解意图并给出相关的回答。</p><h4 id="助理-assistant" tabindex="-1"><a class="header-anchor" href="#助理-assistant"><span>助理（Assistant）</span></a></h4><p><strong>解释：</strong> 在LLM的上下文中，“助理”通常指的是构建在大型语言模型基础上的应用程序或服务，它能够理解用户的查询或指令，并生成人类可读的自然语言回应。这种助理能够执行包括回答问题、提供建议、执行特定任务等多种功能。简而言之，助理是一种应用语言模型技术来实现人机交互的工具或服务。</p><p><strong>注意事项：</strong> 用户在与基于LLM的助理交互时，不需要关心后端的复杂配置，但理解助理的工作原理有助于更有效地与之交互。例如，明确、具体的指令或查询可以帮助助理更准确地理解意图并提供相关的回答或服务。用户也应当了解，虽然助理能提供帮助和信息，但它们的回答可能受限于训练数据和当前技术的局限性。</p><h4 id="令牌-token" tabindex="-1"><a class="header-anchor" href="#令牌-token"><span>令牌（Token）</span></a></h4><p><strong>解释：</strong> 令牌（Token）是语言模型处理文本时的基本单位，可以是一个词、一个字符或者文本的一部分。语言模型根据令牌来理解和生成文本。</p><p><strong>用户设置注意事项：</strong> 用户通常不直接与令牌交互，但应该知道，文本的长度和复杂度可能受到模型处理能力（即令牌限制）的影响。</p><h4 id="上下文长度-context-length" tabindex="-1"><a class="header-anchor" href="#上下文长度-context-length"><span>上下文长度（Context Length）</span></a></h4><p><strong>解释：</strong> 上下文长度是指语言模型在生成响应时能够考虑的最大文本长度（以令牌为单位）。超过这个长度的文本将被忽略。</p><p><strong>注意事项：</strong> 在与语言模型交互时，用户应尽量精简自己的输入，避免超出模型的上下文长度限制，以确保重要信息能被模型考虑。</p><h4 id="温度-temperature" tabindex="-1"><a class="header-anchor" href="#温度-temperature"><span>温度（Temperature）</span></a></h4><p><strong>解释：</strong> 温度是一个控制语言模型生成文本多样性的参数。温度值越高，生成的文本越随机和多样；温度值越低，文本越倾向于常见或预测性强的表达。</p><p><strong>注意事项：</strong> 用户可以根据需要调整温度值以获取更创造性或更精确的回答。但要注意，过高的温度可能导致回答偏离主题或不相关。</p><h4 id="top-k" tabindex="-1"><a class="header-anchor" href="#top-k"><span>Top K</span></a></h4><p><strong>解释：</strong> Top K 是一种文本生成策略，它限制语言模型在每一步生成时只从概率最高的 K 个选项中选择下一个词。</p><p><strong>注意事项：</strong> 调整 Top K 的值可以影响回答的多样性和准确性。较小的 K 值可能使得生成的文本过于常规，而较大的 K 值可能增加文本的多样性但降低准确性。</p><h4 id="top-p" tabindex="-1"><a class="header-anchor" href="#top-p"><span>Top P</span></a></h4><p><strong>解释：</strong> Top P（又称作核采样）是一种文本生成策略，它选择累积概率超过 P 的最小词集，然后从这个集合中随机选择下一个词。</p><p><strong>注意事项：</strong> 调整 Top P 的值可以平衡回答的创新性和相关性。与 Top K 类似，选择合适的 P 值对于生成质量和多样性之间的平衡非常重要。</p><h4 id="min-p" tabindex="-1"><a class="header-anchor" href="#min-p"><span>Min P</span></a></h4><p><strong>解释：</strong> Min P 是在使用 Top P 采样策略时，设置的一个阈值，用来过滤掉概率低于某个特定值 P 的词，以减少生成低概率词的机会。</p><p><strong>注意事项：</strong> 设置 Min P 值可以帮助减少生成不相关或无意义内容的风险。通过调整该值，用户可以控制语言模型输出的质量和一致性。</p><h4 id="重复罚分-repeat-penalty" tabindex="-1"><a class="header-anchor" href="#重复罚分-repeat-penalty"><span>重复罚分（Repeat Penalty）</span></a></h4><p><strong>解释：</strong> 重复罚分是一个避免语言模型重复同一词汇或短语的机制。当检测到重复生成时，会对这些词汇或短语的生成概率进行惩罚，降低它们再次被选中的机率。</p><p><strong>注意事项：</strong> 通过调整重复罚分的大小，用户可以控制生成文本的多样性和新颖性。适当的罚分有助于提高文本的可读性和信息丰富度。</p><h2 id="tbd" tabindex="-1"><a class="header-anchor" href="#tbd"><span>TBD</span></a></h2><hr><p>最后，如果你对本地LLM非常熟悉，欢迎加入我们的讨论！</p><p>Github：https://github.com/xue160709/Local-LLM-User-Guideline</p><p>Discord：https://discord.gg/4AQuf2ctav</p><p>Twitter：https://twitter.com/XueZhirong</p>',303);function y(Q,I){const n=a("ExternalLinkIcon");return o(),r("div",null,[C,s("p",null,[s("a",U,[p("原文"),e(n)])]),R])}const X=t(v,[["render",y],["__file","llm.html.vue"]]),F=JSON.parse('{"path":"/posts/tutorial/start/llm.html","title":"本地LLM使用指南 0.3","lang":"en-US","frontmatter":{"description":"本地LLM使用指南 0.3 原文 作者：XueZhirong 更新日期 2024-3-03：0.3版本增加了常用术语的介绍。 2024-2-24：0.2版本增加了Gemma和Llava模型介绍，增加了GPT4ALL软件介绍，增加了LLM选型介绍。 2024-2-17：0.1版本。 1. 背景介绍 1.1 什么是LLMs？ LLMs，即大型语言模型（La...","head":[["meta",{"property":"og:url","content":"https://llama3.fun/posts/tutorial/start/llm.html"}],["meta",{"property":"og:site_name","content":"Llama3 中文爱好者社区"}],["meta",{"property":"og:title","content":"本地LLM使用指南 0.3"}],["meta",{"property":"og:description","content":"本地LLM使用指南 0.3 原文 作者：XueZhirong 更新日期 2024-3-03：0.3版本增加了常用术语的介绍。 2024-2-24：0.2版本增加了Gemma和Llava模型介绍，增加了GPT4ALL软件介绍，增加了LLM选型介绍。 2024-2-17：0.1版本。 1. 背景介绍 1.1 什么是LLMs？ LLMs，即大型语言模型（La..."}],["meta",{"property":"og:type","content":"article"}],["meta",{"property":"og:locale","content":"en-US"}],["meta",{"property":"og:updated_time","content":"2024-05-01T03:49:24.000Z"}],["meta",{"property":"article:modified_time","content":"2024-05-01T03:49:24.000Z"}],["script",{"type":"application/ld+json"},"{\\"@context\\":\\"https://schema.org\\",\\"@type\\":\\"Article\\",\\"headline\\":\\"本地LLM使用指南 0.3\\",\\"image\\":[\\"\\"],\\"dateModified\\":\\"2024-05-01T03:49:24.000Z\\",\\"author\\":[]}"]]},"headers":[{"level":3,"title":"更新日期","slug":"更新日期","link":"#更新日期","children":[]},{"level":2,"title":"1. 背景介绍","slug":"_1-背景介绍","link":"#_1-背景介绍","children":[{"level":3,"title":"1.1 什么是LLMs？","slug":"_1-1-什么是llms","link":"#_1-1-什么是llms","children":[]},{"level":3,"title":"1.2 LLMs和应用程序的差异","slug":"_1-2-llms和应用程序的差异","link":"#_1-2-llms和应用程序的差异","children":[]},{"level":3,"title":"1.3 开源LLMs有哪些优缺点？","slug":"_1-3-开源llms有哪些优缺点","link":"#_1-3-开源llms有哪些优缺点","children":[]},{"level":3,"title":"1.4 线上和本地LLMs的区别","slug":"_1-4-线上和本地llms的区别","link":"#_1-4-线上和本地llms的区别","children":[]},{"level":3,"title":"1.5 使用线上LLMs存在哪些问题？","slug":"_1-5-使用线上llms存在哪些问题","link":"#_1-5-使用线上llms存在哪些问题","children":[]}]},{"level":2,"title":"2. 哪些理论场景适合使用本地开源LLMs？","slug":"_2-哪些理论场景适合使用本地开源llms","link":"#_2-哪些理论场景适合使用本地开源llms","children":[]},{"level":2,"title":"3. 开箱即用的本地LLMs","slug":"_3-开箱即用的本地llms","link":"#_3-开箱即用的本地llms","children":[{"level":3,"title":"本地LLMs推荐及下载","slug":"本地llms推荐及下载","link":"#本地llms推荐及下载","children":[]},{"level":3,"title":"如何开启本地LLM","slug":"如何开启本地llm","link":"#如何开启本地llm","children":[]},{"level":3,"title":"可连接本地LLM的应用推荐","slug":"可连接本地llm的应用推荐","link":"#可连接本地llm的应用推荐","children":[]}]},{"level":2,"title":"4. 使用本地LLMs需要注意什么？","slug":"_4-使用本地llms需要注意什么","link":"#_4-使用本地llms需要注意什么","children":[]},{"level":2,"title":"5. 常见专业术语","slug":"_5-常见专业术语","link":"#_5-常见专业术语","children":[]},{"level":2,"title":"TBD","slug":"tbd","link":"#tbd","children":[]}],"git":{"updatedTime":1714535364000,"contributors":[{"name":"shadowcz007","email":"chizhiwei007@163.com","commits":1}]},"filePathRelative":"posts/tutorial/start/llm.md","autoDesc":true,"excerpt":"\\n<p><a href=\\"https://github.com/xue160709/Local-LLM-User-Guideline/blob/main/README-zh.md\\" target=\\"_blank\\" rel=\\"noopener noreferrer\\">原文</a></p>\\n<p>作者：XueZhirong</p>\\n<h3>更新日期</h3>\\n<p><strong>2024-3-03</strong>：0.3版本增加了常用术语的介绍。</p>\\n<p><strong>2024-2-24</strong>：0.2版本增加了Gemma和Llava模型介绍，增加了GPT4ALL软件介绍，增加了LLM选型介绍。</p>"}');export{X as comp,F as data};
