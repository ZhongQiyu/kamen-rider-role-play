const express = require('express');
const multer = require('multer');
const pdf = require('pdf-parse');
const cors = require('cors');

const app = express();
const upload = multer({ dest: 'uploads/' });

app.use(cors());

app.post('/upload', upload.array('files'), async (req, res) => {
  const files = req.files;
  let text = '';

  for (const file of files) {
    const data = await pdf(file.path);
    text += data.text;
  }

  const cleanedText = cleanText(text);
  const graphData = buildKnowledgeGraph(cleanedText);

  res.json(graphData);
});

const cleanText = (text) => {
  text = text.replace(/\s+/g, ' ');
  const segments = text.split('华侨大学计算机学院');
  return segments.join('\n');
};

const buildKnowledgeGraph = (text) => {
  const knowledgePoints = [];

  if (text.includes('贝叶斯决策理论')) {
    knowledgePoints.push({ source: '模式识别', target: '贝叶斯决策理论' });
    if (text.includes('贝叶斯公式')) {
      knowledgePoints.push({ source: '贝叶斯决策理论', target: '贝叶斯公式' });
    }
    if (text.includes('先验概率')) {
      knowledgePoints.push({ source: '贝叶斯决策理论', target: '先验概率' });
    }
    if (text.includes('后验概率')) {
      knowledgePoints.push({ source: '贝叶斯决策理论', target: '后验概率' });
    }
    if (text.includes('最大后验估计')) {
      knowledgePoints.push({ source: '贝叶斯决策理论', target: '最大后验估计' });
    }
    if (text.includes('最小错误率决策')) {
      knowledgePoints.push({ source: '贝叶斯决策理论', target: '最小错误率决策' });
    }
    if (text.includes('最小风险决策')) {
      knowledgePoints.push({ source: '贝叶斯决策理论', target: '最小风险决策' });
    }
  }

  if (text.includes('概率密度函数估计')) {
    knowledgePoints.push({ source: '模式识别', target: '概率密度函数估计' });
    if (text.includes('参数估计')) {
      knowledgePoints.push({ source: '概率密度函数估计', target: '参数估计' });
      if (text.includes('矩估计')) {
        knowledgePoints.push({ source: '参数估计', target: '矩估计' });
      }
      if (text.includes('最大似然估计')) {
        knowledgePoints.push({ source: '参数估计', target: '最大似然估计' });
      }
      if (text.includes('贝叶斯估计')) {
        knowledgePoints.push({ source: '参数估计', target: '贝叶斯估计' });
      }
    }
    if (text.includes('非参数估计')) {
      knowledgePoints.push({ source: '概率密度函数估计', target: '非参数估计' });
      if (text.includes('Parzen窗法')) {
        knowledgePoints.push({ source: '非参数估计', target: 'Parzen窗法' });
      }
      if (text.includes('k-近邻法')) {
        knowledgePoints.push({ source: '非参数估计', target: 'k-近邻法' });
      }
    }
  }

  if (text.includes('特征选择和变换')) {
    knowledgePoints.push({ source: '模式识别', target: '特征选择和变换' });
    if (text.includes('特征提取')) {
      knowledgePoints.push({ source: '特征选择和变换', target: '特征提取' });
      if (text.includes('降维')) {
        knowledgePoints.push({ source: '特征提取', target: '降维' });
      }
    }
    if (text.includes('特征选择')) {
      knowledgePoints.push({ source: '特征选择和变换', target: '特征选择' });
      if (text.includes('维数灾难')) {
        knowledgePoints.push({ source: '特征选择', target: '维数灾难' });
      }
    }
  }

  if (text.includes('聚类分析')) {
    knowledgePoints.push({ source: '模式识别', target: '聚类分析' });
    if (text.includes('K-均值算法')) {
      knowledgePoints.push({ source: '聚类分析', target: 'K-均值算法' });
    }
    if (text.includes('层次聚类')) {
      knowledgePoints.push({ source: '聚类分析', target: '层次聚类' });
    }
    if (text.includes('相似性测度')) {
      knowledgePoints.push({ source: '聚类分析', target: '相似性测度' });
    }
  }

  const nodes = [];
  const links = [];
  const uniqueNodes = new Set();

  knowledgePoints.forEach(point => {
    uniqueNodes.add(point.source);
    uniqueNodes.add(point.target);
    links.push({ source: point.source, target: point.target });
  });

  uniqueNodes.forEach(node => {
    nodes.push({ id: node });
  });

  return { nodes, links };
};

// 修改这里的端口为 3000
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
