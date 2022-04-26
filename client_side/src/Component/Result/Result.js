import React from 'react'
import Card from '@mui/material/Card';
import CardActions from '@mui/material/CardActions';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Brightness1RoundedIcon from '@mui/icons-material/Brightness1Rounded';
import Button from '@mui/material/Button';
import { BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './Result.css';
import { styled } from '@mui/material/styles';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell, { tableCellClasses } from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';

const Result = ({ hidden, scoreResult, isEmpty, chartData, chart, chartLength, table }) => {

  function Score({ scoreResult }) {
    switch (scoreResult) {
      case null:
        return <></>
      default:
        return <BasicCard score={scoreResult} />
    }
  }

  function BasicCard({ score }) {
    let val
    if (score >= 2) {
      val = <Typography variant="h6" component="div" color="#079615">
        Score de cohérence : {score}
      </Typography>
    }
    else if (score >= 1) {
      val = <Typography variant="h6" component="div" color="#FF9A02">
        Score de cohérence : {score}
      </Typography>
    }
    else if (score >= 0) {
      val = <Typography variant="h6" component="div" color="#E33A3A">
        Score de cohérence : {score}
      </Typography>
    }
    else {
      val = <Typography variant="h6" component="div">
        Score de coherence : {score}
      </Typography>
    }
    return (
      <div className='result'>
        <Card sx={{ minWidth: 275, border: 1 }}>
          <CardContent>
            <Typography variant="h6" component="div">
              {val}
            </Typography>
          </CardContent>
          <CardActions sx={{ position: 'relative' }}>
            <Button size="small">Voir plus</Button>
          </CardActions>
        </Card>
      </div>
    );
  }

  function RenderChart({ isEmpty, chartData }) {
    if (isEmpty === true) {
      return <></>
    }
    else {
      return (
        <>

          <div style={{ display: 'block' }}>
            <div style={{ display: 'flex', justifyContent: 'center' }}>
              <BarChart
                width={500}
                height={300}
                data={chartData}
                margin={{
                  top: 0,
                  right: 30,
                  left: 20,
                  bottom: 5,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="label" />
                <YAxis tickCount={chartLength} />
                <Tooltip />
                <Legend />
                <Bar dataKey="score" fill="#ffab00" />
              </BarChart>
            </div>
            <div>
              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', marginBottom: '50px' }} >
                <Typography variant='body2' sx={{ fontSize: 18, fontFamily: 'Didact Gothic' }} color="#000" gutterBottom>
                  Nombre de documents par classe de cohérence
                </Typography>
              </div>
            </div>
          </div>
          <RenderTable rows={table}></RenderTable>

        </>

      )
    }
  }

  function Render({ chart }) {
    if (chart === true) {
      return (
        <>
          <div>
            <RenderChart isEmpty={isEmpty} chartData={chartData} />
          </div>

        </>
      )
    }
    else {
      return (
        <>
          <Score scoreResult={scoreResult} />
          <div className='cards-key'>
            <div className='card1'>
              <div><Brightness1RoundedIcon sx={{ color: "#079615" }} /></div>
              <p id='scoreCard'>2 - 3 (élevé)</p>
            </div>
            <div className='card1'>
              <div><Brightness1RoundedIcon sx={{ color: "#FF9A02" }} /></div>
              <p id='scoreCard'>1 - 2 (moyen)</p>
            </div>
            <div className='card1'>
              <div><Brightness1RoundedIcon sx={{ color: "#E33A3A" }} /></div>
              <p id='scoreCard'>0 - 1 (bas)</p>
            </div>
          </div>
        </>
      )
    }
  }

  const StyledTableCell = styled(TableCell)(({ theme }) => ({
    [`&.${tableCellClasses.head}`]: {
      backgroundColor: "#0288d1",
      color: theme.palette.common.white,
    },
    [`&.${tableCellClasses.body}`]: {
      fontSize: 14,
    },
  }));

  const StyledTableRow = styled(TableRow)(({ theme }) => ({
    '&:nth-of-type(odd)': {
      backgroundColor: theme.palette.action.hover,
    },
    // hide last border
    '&:last-child td, &:last-child th': {
      border: 0,
    },
  }));

  function RenderTable({ rows }) {
    return (
      <div>
        <Table sx={{ minWidth: 70, maxWidth: 1000 }} aria-label="customized table">
          <TableHead>
            <TableRow>
              <StyledTableCell >ID du document</StyledTableCell>
              <StyledTableCell align="center">Texte</StyledTableCell>
              <StyledTableCell align="left">Score original</StyledTableCell>
              <StyledTableCell align="left">Score prédit</StyledTableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map((row) => (
              <StyledTableRow key={row.text_id}>
                <StyledTableCell component="th" scope="row">
                  {row.text_id}
                </StyledTableCell>
                <StyledTableCell align="left">{row.text}</StyledTableCell>
                <StyledTableCell align="left">{row.original_score}</StyledTableCell>
                <StyledTableCell align="left" sx={{ fontWeight: "bold" }}>{row.predicted_score}</StyledTableCell>
              </StyledTableRow>
            ))}
          </TableBody>
        </Table>
      </div>

    );
  }
  return (
    <div id='evalSection' hidden={hidden}>
      <Render chart={chart} />
    </div>
  )
}

export default Result