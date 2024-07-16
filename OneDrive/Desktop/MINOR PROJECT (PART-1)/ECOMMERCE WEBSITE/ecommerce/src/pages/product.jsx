import React, { useContext } from 'react'
import { ShopContext } from '../context/shopcontext'
import {useParams} from'react-router-dom'
import Breadcrums from '../components/breadcrums/breadcrums';
import Productdisplay from '../components/productdisplay/productdisplay';
import Descriptionbox from '../components/descriptionbox/descriptionbox';
import Relatedproducts from '../components/relatedproducts/relatedproducts';
export default function Product() {
  const {all_product}=useContext(ShopContext);
  const {productId}=useParams();
  const product=all_product.find((e)=>e.id=== Number(productId))
  return (
    <div>
      {/* <Breadcrums product={product}></Breadcrums>
       */}
      <Breadcrums product={product}></Breadcrums>
      <Productdisplay product={product}></Productdisplay>
      <Descriptionbox></Descriptionbox>
      <Relatedproducts></Relatedproducts>
    </div>
  )
}
